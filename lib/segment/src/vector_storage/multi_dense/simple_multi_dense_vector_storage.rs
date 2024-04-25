use std::borrow::Cow;
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use bitvec::prelude::{BitSlice, BitVec};
use common::types::PointOffsetType;
use parking_lot::RwLock;
use rocksdb::DB;

use crate::common::operation_error::{check_process_stopped, OperationError, OperationResult};
use crate::common::rocksdb_wrapper::DatabaseColumnWrapper;
use crate::common::Flusher;
use crate::data_types::named_vectors::CowVector;
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::{
    MultiDenseVector, TypedMultiDenseVector, TypedMultiDenseVectorRef, VectorElementType, VectorRef,
};
use crate::types::{Distance, MultiVectorConfig, VectorStorageDatatype};
use crate::vector_storage::bitvec::bitvec_set_deleted;
use crate::vector_storage::common::StoredRecord;
use crate::vector_storage::{MultiVectorStorage, VectorStorage, VectorStorageEnum};

type StoredMultiDenseVector<T> = StoredRecord<TypedMultiDenseVector<T>>;

/// In-memory vector storage with on-update persistence using `store`
pub struct SimpleMultiDenseVectorStorage<T: PrimitiveVectorElement> {
    dim: usize,
    distance: Distance,
    multi_vector_config: MultiVectorConfig,
    /// Keep vectors in memory
    vectors: Vec<TypedMultiDenseVector<T>>,
    db_wrapper: DatabaseColumnWrapper,
    update_buffer: StoredMultiDenseVector<T>,
    /// BitVec for deleted flags. Grows dynamically upto last set flag.
    deleted: BitVec,
    /// Current number of deleted vectors.
    deleted_count: usize,
}

#[allow(unused)]
pub fn open_simple_multi_dense_vector_storage(
    database: Arc<RwLock<DB>>,
    database_column_name: &str,
    dim: usize,
    distance: Distance,
    multi_vector_config: MultiVectorConfig,
    stopped: &AtomicBool,
) -> OperationResult<Arc<AtomicRefCell<VectorStorageEnum>>> {
    let mut vectors: Vec<TypedMultiDenseVector<VectorElementType>> = vec![];
    let (mut deleted, mut deleted_count) = (BitVec::new(), 0);
    let db_wrapper = DatabaseColumnWrapper::new(database, database_column_name);
    db_wrapper.lock_db().iter()?;
    for (key, value) in db_wrapper.lock_db().iter()? {
        let point_id: PointOffsetType = bincode::deserialize(&key)
            .map_err(|_| OperationError::service_error("cannot deserialize point id from db"))?;
        let stored_record: StoredMultiDenseVector<VectorElementType> = bincode::deserialize(&value)
            .map_err(|_| OperationError::service_error("cannot deserialize record from db"))?;

        // Propagate deleted flag
        if stored_record.deleted {
            bitvec_set_deleted(&mut deleted, point_id, true);
            deleted_count += 1;
        }
        let point_id_usize = point_id as usize;
        if point_id_usize >= vectors.len() {
            vectors.resize(point_id_usize + 1, TypedMultiDenseVector::placeholder(dim));
        }
        vectors[point_id_usize] = stored_record.vector;

        check_process_stopped(stopped)?;
    }

    Ok(Arc::new(AtomicRefCell::new(
        VectorStorageEnum::MultiDenseSimple(SimpleMultiDenseVectorStorage {
            dim,
            distance,
            multi_vector_config,
            vectors,
            db_wrapper,
            update_buffer: StoredMultiDenseVector {
                deleted: false,
                vector: TypedMultiDenseVector::placeholder(dim),
            },
            deleted,
            deleted_count,
        }),
    )))
}

impl<T: PrimitiveVectorElement> SimpleMultiDenseVectorStorage<T> {
    /// Set deleted flag for given key. Returns previous deleted state.
    #[inline]
    fn set_deleted(&mut self, key: PointOffsetType, deleted: bool) -> bool {
        if key as usize >= self.vectors.len() {
            return false;
        }
        let was_deleted = bitvec_set_deleted(&mut self.deleted, key, deleted);
        if was_deleted != deleted {
            if !was_deleted {
                self.deleted_count += 1;
            } else {
                self.deleted_count = self.deleted_count.saturating_sub(1);
            }
        }
        was_deleted
    }

    fn update_stored(
        &mut self,
        key: PointOffsetType,
        deleted: bool,
        vector: Option<TypedMultiDenseVector<T>>,
    ) -> OperationResult<()> {
        // Write vector state to buffer record
        let record = &mut self.update_buffer;
        record.deleted = deleted;
        if let Some(vector) = vector {
            record.vector = vector;
        }

        // Store updated record
        self.db_wrapper.put(
            bincode::serialize(&key).unwrap(),
            bincode::serialize(&record).unwrap(),
        )?;

        Ok(())
    }
}

impl<T: PrimitiveVectorElement> MultiVectorStorage<T> for SimpleMultiDenseVectorStorage<T> {
    fn get_multi(&self, key: PointOffsetType) -> TypedMultiDenseVectorRef<T> {
        TypedMultiDenseVectorRef::from(self.vectors.get(key as usize).expect("vector not found"))
    }

    fn multi_vector_config(&self) -> &MultiVectorConfig {
        &self.multi_vector_config
    }
}

impl<T: PrimitiveVectorElement> VectorStorage for SimpleMultiDenseVectorStorage<T> {
    fn vector_dim(&self) -> usize {
        self.dim
    }

    fn distance(&self) -> Distance {
        self.distance
    }

    fn datatype(&self) -> VectorStorageDatatype {
        VectorStorageDatatype::Float32
    }

    fn is_on_disk(&self) -> bool {
        false
    }

    fn total_vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn get_vector(&self, key: PointOffsetType) -> CowVector {
        let multi_dense_vector = self.vectors.get(key as usize).expect("vector not found");
        let multi_dense_vector = T::into_float_multivector(Cow::Borrowed(multi_dense_vector));
        CowVector::from(multi_dense_vector)
    }

    fn insert_vector(&mut self, key: PointOffsetType, vector: VectorRef) -> OperationResult<()> {
        let vector: &MultiDenseVector = vector.try_into()?;
        let multi_vector = T::from_float_multivector(Cow::Borrowed(vector)).into_owned();
        let key_usize = key as usize;
        if key_usize >= self.vectors.len() {
            self.vectors
                .resize(key_usize + 1, TypedMultiDenseVector::placeholder(self.dim));
        }
        self.vectors[key_usize] = multi_vector.clone();
        self.set_deleted(key, false);
        self.update_stored(key, false, Some(multi_vector))?;
        Ok(())
    }

    fn update_from(
        &mut self,
        other: &VectorStorageEnum,
        other_ids: &mut impl Iterator<Item = PointOffsetType>,
        stopped: &AtomicBool,
    ) -> OperationResult<Range<PointOffsetType>> {
        let start_index = self.vectors.len() as PointOffsetType;
        for point_id in other_ids {
            check_process_stopped(stopped)?;
            // Do not perform preprocessing - vectors should be already processed
            let other_vector = other.get_vector(point_id);
            let other_vector: &TypedMultiDenseVector<VectorElementType> =
                other_vector.as_vec_ref().try_into()?;
            let other_multi_vector =
                T::from_float_multivector(Cow::Borrowed(other_vector)).into_owned();
            let other_deleted = other.is_deleted_vector(point_id);
            self.vectors.push(other_multi_vector.clone());
            let new_id = self.vectors.len() as PointOffsetType - 1;
            self.set_deleted(new_id, other_deleted);
            self.update_stored(new_id, other_deleted, Some(other_multi_vector))?;
        }
        let end_index = self.vectors.len() as PointOffsetType;
        Ok(start_index..end_index)
    }

    fn flusher(&self) -> Flusher {
        self.db_wrapper.flusher()
    }

    fn files(&self) -> Vec<std::path::PathBuf> {
        vec![]
    }

    fn delete_vector(&mut self, key: PointOffsetType) -> OperationResult<bool> {
        let is_deleted = !self.set_deleted(key, true);
        if is_deleted {
            self.update_stored(key, true, None)?;
        }
        Ok(is_deleted)
    }

    fn is_deleted_vector(&self, key: PointOffsetType) -> bool {
        self.deleted.get(key as usize).map(|b| *b).unwrap_or(false)
    }

    fn deleted_vector_count(&self) -> usize {
        self.deleted_count
    }

    fn deleted_vector_bitslice(&self) -> &BitSlice {
        self.deleted.as_bitslice()
    }
}
