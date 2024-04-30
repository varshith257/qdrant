#![allow(unused_imports)]

#[cfg(not(target_os = "windows"))]
mod prof;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use common::cpu::CpuPermit;
use common::types::PointOffsetType;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use segment::fixtures::sparse_fixtures::fixture_sparse_index_ram;
use segment::index::hnsw_index::num_rayon_threads;
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::index::sparse_index::sparse_vector_index::SparseVectorIndex;
use segment::index::{PayloadIndex, VectorIndex};
use segment::types::PayloadSchemaType::Keyword;
use segment::types::{Condition, FieldCondition, Filter, Payload};
use serde_json::json;
use sparse::common::sparse_vector_fixture::random_positive_sparse_vector;
use sparse::index::inverted_index::inverted_index_mmap::InvertedIndexMmap;
use tempfile::Builder;

const NUM_VECTORS: usize = 50_000;
const MAX_SPARSE_DIM: usize = 30_000;
const TOP: usize = 10;
const FULL_SCAN_THRESHOLD: usize = 1; // low value to trigger index usage by default

fn sparse_vector_index_search_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse-vector-search-group");

    let stopped = AtomicBool::new(false);
    let mut rnd = StdRng::seed_from_u64(42);

    let data_dir = Builder::new().prefix("data_dir").tempdir().unwrap();
    let sparse_vector_index = fixture_sparse_index_ram(
        &mut rnd,
        NUM_VECTORS,
        MAX_SPARSE_DIM,
        FULL_SCAN_THRESHOLD,
        data_dir.path(),
        &stopped,
    );

    // adding payload on field
    let field_name = "field";
    let field_value = "important value";
    let payload: Payload = json!({
        field_name: field_value,
    })
    .into();

    // all points have the same payload
    let mut payload_index = sparse_vector_index.payload_index.borrow_mut();
    for idx in 0..NUM_VECTORS {
        payload_index
            .assign(idx as PointOffsetType, &payload, &None)
            .unwrap();
    }
    drop(payload_index);

    // shared query vector (positive values to test pruning)
    let vector = random_positive_sparse_vector(&mut rnd, MAX_SPARSE_DIM);
    eprintln!("sparse_vector size = {:#?}", vector.values.len());
    let sparse_vector = vector.clone();
    let query_vector = vector.into();

    let permit_cpu_count = num_rayon_threads(0);
    #[cfg(any())]
    let permit = Arc::new(CpuPermit::dummy(permit_cpu_count as u32));

    // mmap inverted index
    #[cfg(any())]
    let mmap_index_dir = Builder::new().prefix("mmap_index_dir").tempdir().unwrap();
    #[cfg(any())]
    let sparse_index_config =
        SparseIndexConfig::new(Some(FULL_SCAN_THRESHOLD), SparseIndexType::Mmap);
    #[cfg(any())]
    let mut sparse_vector_index_mmap: SparseVectorIndex<InvertedIndexMmap> =
        SparseVectorIndex::open(
            sparse_index_config,
            sparse_vector_index.id_tracker.clone(),
            sparse_vector_index.vector_storage.clone(),
            sparse_vector_index.payload_index.clone(),
            mmap_index_dir.path(),
            &stopped,
        )
        .unwrap();
    #[cfg(any())]
    sparse_vector_index_mmap
        .build_index(permit, &stopped)
        .unwrap();
    #[cfg(any())]
    assert_eq!(sparse_vector_index_mmap.indexed_vector_count(), NUM_VECTORS);

    // intent: bench `search` without filter on mmap inverted index
    // disabled
    #[cfg(any())]
    group.bench_function("mmap-inverted-index-search", |b| {
        b.iter(|| {
            let results = sparse_vector_index_mmap
                .search(
                    &[&query_vector],
                    None,
                    TOP,
                    None,
                    &stopped,
                    &Default::default(),
                )
                .unwrap();

            assert_eq!(results[0].len(), TOP);
        })
    });

    // intent: bench `search` without filter
    group.bench_function("inverted-index-search", |b| {
        b.iter(|| {
            let results = sparse_vector_index
                .search(
                    &[&query_vector],
                    None,
                    TOP,
                    None,
                    &stopped,
                    &Default::default(),
                )
                .unwrap();

            assert_eq!(results[0].len(), TOP);
        })
    });

    // filter by field
    let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
        field_name.parse().unwrap(),
        field_value.to_owned().into(),
    )));

    // intent: bench plain search when the filtered payload key is not indexed
    group.bench_function("inverted-index-filtered-plain", |b| {
        b.iter(|| {
            let mut prefiltered_points = None;
            let results = sparse_vector_index
                .search_plain(
                    &sparse_vector,
                    &filter,
                    TOP,
                    &stopped,
                    &mut prefiltered_points,
                )
                .unwrap();

            assert_eq!(results.len(), TOP);
        })
    });

    let mut payload_index = sparse_vector_index.payload_index.borrow_mut();

    // create payload field index
    payload_index
        .set_indexed(&field_name.parse().unwrap(), Keyword.into())
        .unwrap();

    drop(payload_index);

    // intent: bench `search` when the filtered payload key is indexed
    group.bench_function("inverted-index-filtered-payload-index", |b| {
        b.iter(|| {
            let results = sparse_vector_index
                .search(
                    &[&query_vector],
                    Some(&filter),
                    TOP,
                    None,
                    &stopped,
                    &Default::default(),
                )
                .unwrap();

            assert_eq!(results[0].len(), TOP);
        })
    });

    // intent: bench plain search when the filtered payload key is indexed
    group.bench_function("plain-filtered-payload-index", |b| {
        b.iter(|| {
            let mut prefiltered_points = None;
            let results = sparse_vector_index
                .search_plain(
                    &sparse_vector,
                    &filter,
                    TOP,
                    &stopped,
                    &mut prefiltered_points,
                )
                .unwrap();

            assert_eq!(results.len(), TOP);
        })
    });

    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    // config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    config = Criterion::default();
    targets = sparse_vector_index_search_benchmark
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = sparse_vector_index_search_benchmark,
}

criterion_main!(benches);
