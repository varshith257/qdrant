use std::collections::{HashMap, HashSet};
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use common::cpu::CpuPermit;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use itertools::Itertools;
use sparse::common::scores_memory_pool::ScoresMemoryPool;
use sparse::common::sparse_vector::SparseVector;
use sparse::common::types::DimId;
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::search_context::SearchContext;

use super::indices_tracker::IndicesTracker;
use super::sparse_index_config::SparseIndexType;
use crate::common::operation_error::{check_process_stopped, OperationError, OperationResult};
use crate::common::operation_time_statistics::ScopeDurationMeasurer;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, Vector, VectorRef};
use crate::id_tracker::IdTrackerSS;
use crate::index::field_index::CardinalityEstimation;
use crate::index::query_estimator::adjust_to_available_vectors;
use crate::index::sparse_index::sparse_index_config::SparseIndexConfig;
use crate::index::sparse_index::sparse_search_telemetry::SparseSearchesTelemetry;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::index::{PayloadIndex, VectorIndex};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Filter, SearchParams, DEFAULT_SPARSE_FULL_SCAN_THRESHOLD};
use crate::vector_storage::query::TransformInto;
use crate::vector_storage::{
    check_deleted_condition, new_stoppable_raw_scorer, VectorStorage, VectorStorageEnum,
};

pub struct SparseVectorIndex<TInvertedIndex: InvertedIndex> {
    pub config: SparseIndexConfig,
    pub id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
    pub vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    pub payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    path: PathBuf,
    pub inverted_index: TInvertedIndex,
    searches_telemetry: SparseSearchesTelemetry,
    is_appendable: bool,
    pub indices_tracker: IndicesTracker,
    scores_memory_pool: ScoresMemoryPool,
}

impl<TInvertedIndex: InvertedIndex> SparseVectorIndex<TInvertedIndex> {
    /// Open a sparse vector index at a given path
    pub fn open(
        config: SparseIndexConfig,
        id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
        path: &Path,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        // create directory if it does not exist
        create_dir_all(path)?;
        let is_appendable = config.index_type == SparseIndexType::MutableRam;

        let config_path = SparseIndexConfig::get_config_path(path);
        let (config, inverted_index, indices_tracker) = if is_appendable {
            // RAM mutable case - build inverted index from scratch and use provided config
            let (inverted_index, indices_tracker) = Self::build_inverted_index(
                id_tracker.clone(),
                vector_storage.clone(),
                path,
                stopped,
            )?;
            (config, inverted_index, indices_tracker)
        } else if config_path.exists() {
            // Load inverted index and config
            let loaded_config = SparseIndexConfig::load(&config_path)?;
            let inverted_index = TInvertedIndex::open(path)?;
            let indices_tracker =
                IndicesTracker::open(path, || inverted_index.max_index().unwrap_or_default())?;
            (loaded_config, inverted_index, indices_tracker)
        } else {
            // Inverted index and config are not presented - initialize empty inverted index
            let inverted_index = TInvertedIndex::from_ram_index(InvertedIndexRam::empty(), path)?;
            let indices_tracker = Default::default();
            (config, inverted_index, indices_tracker)
        };

        let searches_telemetry = SparseSearchesTelemetry::new();
        let path = path.to_path_buf();
        let scores_memory_pool = ScoresMemoryPool::new();
        Ok(Self {
            config,
            id_tracker,
            vector_storage,
            payload_index,
            path,
            inverted_index,
            searches_telemetry,
            is_appendable,
            indices_tracker,
            scores_memory_pool,
        })
    }

    fn save_config(&self) -> OperationResult<()> {
        let config_path = SparseIndexConfig::get_config_path(&self.path);
        self.config.save(&config_path)
    }

    fn build_inverted_index(
        id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        path: &Path,
        stopped: &AtomicBool,
    ) -> OperationResult<(TInvertedIndex, IndicesTracker)> {
        let borrowed_vector_storage = vector_storage.borrow();
        let borrowed_id_tracker = id_tracker.borrow();
        let deleted_bitslice = borrowed_vector_storage.deleted_vector_bitslice();

        let mut ram_index_builder = InvertedIndexBuilder::new();
        let mut indices_tracker = IndicesTracker::default();
        for id in borrowed_id_tracker.iter_ids_excluding(deleted_bitslice) {
            check_process_stopped(stopped)?;
            // It is possible that the vector is not present in the storage in case of crash.
            // Because:
            // - the `id_tracker` is flushed before the `vector_storage`
            // - the sparse index is built *before* recovering the WAL when loading a segment
            match borrowed_vector_storage.get_vector_opt(id) {
                None => {
                    // the vector was lost in a crash but will be recovered by the WAL
                    log::debug!("Sparse vector with id {} is not found", id)
                }
                Some(vector) => {
                    let vector: &SparseVector = vector.as_vec_ref().try_into()?;
                    // do not index empty vectors
                    if vector.is_empty() {
                        continue;
                    }
                    indices_tracker.register_indices(vector);
                    let vector = indices_tracker.remap_vector(vector.to_owned());
                    ram_index_builder.add(id, vector);
                }
            }
        }
        Ok((
            TInvertedIndex::from_ram_index(ram_index_builder.build(), path)?,
            indices_tracker,
        ))
    }

    /// Returns the maximum number of results that can be returned by the index for a given sparse vector
    /// Warning: the cost of this function grows with the number of dimensions in the query vector
    pub fn max_result_count(&self, query_vector: &SparseVector) -> usize {
        let mut unique_record_ids = HashSet::new();
        for dim_id in query_vector.indices.iter() {
            if let Some(dim_id) = self.indices_tracker.remap_index(*dim_id) {
                if let Some(posting_list) = self.inverted_index.get(&dim_id) {
                    for element in posting_list {
                        unique_record_ids.insert(element.record_id);
                    }
                }
            }
        }
        unique_record_ids.len()
    }

    fn get_query_cardinality(&self, filter: &Filter) -> CardinalityEstimation {
        let vector_storage = self.vector_storage.borrow();
        let id_tracker = self.id_tracker.borrow();
        let payload_index = self.payload_index.borrow();
        let available_vector_count = vector_storage.available_vector_count();
        let query_point_cardinality = payload_index.estimate_cardinality(filter);
        adjust_to_available_vectors(
            query_point_cardinality,
            available_vector_count,
            id_tracker.available_point_count(),
        )
    }

    // Search using raw scorer
    fn search_scored(
        &self,
        query_vector: &QueryVector,
        filter: Option<&Filter>,
        top: usize,
        is_stopped: &AtomicBool,
        prefiltered_points: &mut Option<Vec<PointOffsetType>>,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        let vector_storage = self.vector_storage.borrow();
        let id_tracker = self.id_tracker.borrow();
        let raw_scorer = new_stoppable_raw_scorer(
            query_vector.clone(),
            &vector_storage,
            id_tracker.deleted_point_bitslice(),
            is_stopped,
        )?;
        match filter {
            Some(filter) => {
                let payload_index = self.payload_index.borrow();
                let mut filtered_points = match prefiltered_points {
                    Some(filtered_points) => filtered_points.iter().copied(),
                    None => {
                        let filtered_points = payload_index.query_points(filter);
                        *prefiltered_points = Some(filtered_points);
                        prefiltered_points.as_ref().unwrap().iter().copied()
                    }
                };
                Ok(raw_scorer.peek_top_iter(&mut filtered_points, top))
            }
            None => Ok(raw_scorer.peek_top_all(top)),
        }
    }

    pub fn search_plain(
        &self,
        sparse_vector: &SparseVector,
        filter: &Filter,
        top: usize,
        is_stopped: &AtomicBool,
        prefiltered_points: &mut Option<Vec<PointOffsetType>>,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        let vector_storage = self.vector_storage.borrow();
        let id_tracker = self.id_tracker.borrow();
        let payload_index = self.payload_index.borrow();

        let deleted_point_bitslice = id_tracker.deleted_point_bitslice();
        let deleted_vectors = vector_storage.deleted_vector_bitslice();

        let ids = match prefiltered_points {
            Some(filtered_points) => filtered_points.iter(),
            None => {
                let filtered_points = payload_index.query_points(filter);
                *prefiltered_points = Some(filtered_points);
                prefiltered_points.as_ref().unwrap().iter()
            }
        }
        .copied()
        .filter(|&idx| check_deleted_condition(idx, deleted_vectors, deleted_point_bitslice))
        .collect_vec();

        let sparse_vector = self.indices_tracker.remap_vector(sparse_vector.to_owned());
        let memory_handle = self.scores_memory_pool.get();
        let mut search_context = SearchContext::new(
            sparse_vector,
            top,
            &self.inverted_index,
            memory_handle,
            is_stopped,
        );
        Ok(search_context.plain_search(&ids))
    }

    // search using sparse vector inverted index
    fn search_sparse(
        &self,
        sparse_vector: &SparseVector,
        filter: Option<&Filter>,
        top: usize,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        let vector_storage = self.vector_storage.borrow();
        let id_tracker = self.id_tracker.borrow();
        let deleted_point_bitslice = id_tracker.deleted_point_bitslice();
        let deleted_vectors = vector_storage.deleted_vector_bitslice();

        let not_deleted_condition = |idx: PointOffsetType| -> bool {
            check_deleted_condition(idx, deleted_vectors, deleted_point_bitslice)
        };
        let sparse_vector = self.indices_tracker.remap_vector(sparse_vector.to_owned());
        let memory_handle = self.scores_memory_pool.get();
        let mut search_context = SearchContext::new(
            sparse_vector,
            top,
            &self.inverted_index,
            memory_handle,
            is_stopped,
        );

        match filter {
            Some(filter) => {
                let payload_index = self.payload_index.borrow();
                let filter_context = payload_index.filter_context(filter);
                let matches_filter_condition = |idx: PointOffsetType| -> bool {
                    not_deleted_condition(idx) && filter_context.check(idx)
                };
                Ok(search_context.search(&matches_filter_condition))
            }
            None => Ok(search_context.search(&not_deleted_condition)),
        }
    }

    fn search_nearest_query(
        &self,
        vector: &SparseVector,
        filter: Option<&Filter>,
        top: usize,
        is_stopped: &AtomicBool,
        prefiltered_points: &mut Option<Vec<PointOffsetType>>,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if vector.is_empty() {
            return Ok(vec![]);
        }
        let mut vector = vector.clone();
        vector.sort_by_indices();

        match filter {
            Some(filter) => {
                // if cardinality is small - use plain search
                let query_cardinality = self.get_query_cardinality(filter);
                let threshold = self
                    .config
                    .full_scan_threshold
                    .unwrap_or(DEFAULT_SPARSE_FULL_SCAN_THRESHOLD);
                if query_cardinality.max < threshold {
                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.small_cardinality);
                    self.search_plain(&vector, filter, top, is_stopped, prefiltered_points)
                } else {
                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.filtered_sparse);
                    self.search_sparse(&vector, Some(filter), top, is_stopped)
                }
            }
            None => {
                let _timer = ScopeDurationMeasurer::new(&self.searches_telemetry.unfiltered_sparse);
                self.search_sparse(&vector, filter, top, is_stopped)
            }
        }
    }

    pub fn search_query(
        &self,
        query_vector: &QueryVector,
        filter: Option<&Filter>,
        top: usize,
        is_stopped: &AtomicBool,
        prefiltered_points: &mut Option<Vec<PointOffsetType>>,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if top == 0 {
            return Ok(vec![]);
        }

        match query_vector {
            QueryVector::Nearest(vector) => self.search_nearest_query(
                vector.try_into()?,
                filter,
                top,
                is_stopped,
                prefiltered_points,
            ),
            QueryVector::Recommend(_) | QueryVector::Discovery(_) | QueryVector::Context(_) => {
                let _timer = if filter.is_some() {
                    ScopeDurationMeasurer::new(&self.searches_telemetry.filtered_plain)
                } else {
                    ScopeDurationMeasurer::new(&self.searches_telemetry.unfiltered_plain)
                };
                self.search_scored(query_vector, filter, top, is_stopped, prefiltered_points)
            }
        }
    }

    // Update statistics for idf-dot similarity
    pub fn fill_idf_statistics(&self, idf: &mut HashMap<DimId, usize>) {
        for (dim_id, count) in idf.iter_mut() {
            if let Some(remapped_dim_id) = self.indices_tracker.remap_index(*dim_id) {
                if let Some(posting_list_len) =
                    self.inverted_index.posting_list_len(&remapped_dim_id)
                {
                    *count = posting_list_len
                }
            }
        }
    }
}

impl<TInvertedIndex: InvertedIndex> VectorIndex for SparseVectorIndex<TInvertedIndex> {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        _params: Option<&SearchParams>,
        is_stopped: &AtomicBool,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let mut results = Vec::with_capacity(vectors.len());
        let mut prefiltered_points = None;
        for vector in vectors {
            check_process_stopped(is_stopped)?;

            let search_results = if query_context.is_require_idf() {
                let vector = (*vector).clone().transform(|mut vector| {
                    match &mut vector {
                        Vector::Dense(_) | Vector::MultiDense(_) => {
                            return Err(OperationError::WrongSparse);
                        }
                        Vector::Sparse(sparse) => {
                            query_context.remap_idf_weights(&sparse.indices, &mut sparse.values)
                        }
                    }

                    Ok(vector)
                })?;

                self.search_query(&vector, filter, top, is_stopped, &mut prefiltered_points)?
            } else {
                self.search_query(vector, filter, top, is_stopped, &mut prefiltered_points)?
            };

            results.push(search_results);
        }
        Ok(results)
    }

    fn build_index(
        &mut self,
        _permit: Arc<CpuPermit>,
        stopped: &AtomicBool,
    ) -> OperationResult<()> {
        let (inverted_index, indices_tracker) = Self::build_inverted_index(
            self.id_tracker.clone(),
            self.vector_storage.clone(),
            &self.path,
            stopped,
        )?;

        self.inverted_index = inverted_index;
        self.indices_tracker = indices_tracker;

        // save inverted index
        if !self.is_appendable {
            self.indices_tracker.save(&self.path)?;
            self.inverted_index.save(&self.path)?;
        }

        // save config to mark successful build
        self.save_config()?;
        Ok(())
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.searches_telemetry.get_telemetry_data(detail)
    }

    fn files(&self) -> Vec<PathBuf> {
        let config_file = SparseIndexConfig::get_config_path(&self.path);
        if !config_file.exists() {
            return vec![];
        }

        let mut all_files = vec![];

        let indices_tracker_file = IndicesTracker::file_path(&self.path);
        if indices_tracker_file.exists() {
            all_files.push(indices_tracker_file);
        }

        all_files.push(config_file);
        all_files.extend_from_slice(&TInvertedIndex::files(&self.path));
        all_files
    }

    fn indexed_vector_count(&self) -> usize {
        self.inverted_index.vector_count()
    }

    fn update_vector(&mut self, id: PointOffsetType, vector: VectorRef) -> OperationResult<()> {
        if !self.is_appendable {
            return Err(OperationError::service_error(
                "Cannot update vector in non-appendable index",
            ));
        }

        let vector: &SparseVector = vector.try_into()?;
        // do not upsert empty vectors into the index
        if !vector.is_empty() {
            self.indices_tracker.register_indices(vector);
            let vector = self.indices_tracker.remap_vector(vector.to_owned());
            self.inverted_index.upsert(id, vector);
        }
        Ok(())
    }
}
