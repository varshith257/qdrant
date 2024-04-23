use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use common::types::PointOffsetType;
use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng as _;
use sparse::common::scores_memory_pool::ScoresMemoryPool;
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::common::sparse_vector_fixture::{random_positive_sparse_vector, random_sparse_vector};
use sparse::index::csr;
use sparse::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use sparse::index::search_context::SearchContext;
mod prof;

const NUM_QUERIES: usize = 2048;
const MAX_SPARSE_DIM: usize = 30_000;
const TOP: usize = 10;

pub fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    bench_search_random(&mut group, "random_50k", 50_000);
    bench_search_random(&mut group, "random_500k", 500_000);

    bench_search_msmarco(&mut group, "msmarco_1M", "base_1M.csr", 1.0);
    bench_search_msmarco(&mut group, "msmarco_full_0.25", "base_full.csr", 0.25);
}

fn bench_search_random<M: Measurement>(c: &mut BenchmarkGroup<M>, name: &str, num_vectors: usize) {
    let mut rnd = StdRng::seed_from_u64(42);

    // index
    let mut builder = InvertedIndexBuilder::new();
    for idx in 0..num_vectors {
        let vec = random_sparse_vector(&mut rnd, MAX_SPARSE_DIM);
        builder.add(
            idx as PointOffsetType,
            RemappedSparseVector::new(vec.indices, vec.values).unwrap(),
        );
    }
    let index = builder.build();

    let query_vectors = (0..NUM_QUERIES)
        .map(|_| {
            let vector = random_positive_sparse_vector(&mut rnd, MAX_SPARSE_DIM);
            RemappedSparseVector::new(vector.indices, vector.values).unwrap()
        })
        .collect::<Vec<_>>();
    let mut it = query_vectors.iter().cycle();

    let pool = ScoresMemoryPool::new();
    let stopped = AtomicBool::new(false);

    c.bench_function(name, |b| {
        b.iter(|| {
            SearchContext::new(
                it.next().unwrap().clone(),
                TOP,
                &index,
                pool.get(),
                &stopped,
            )
            .search(&|_| true)
        })
    });
}

pub fn bench_search_msmarco<M: Measurement>(
    c: &mut BenchmarkGroup<M>,
    name: &str,
    dataset: &str,
    ratio: f32,
) {
    let base_dir = PathBuf::from(std::env::var("MSMARCO_DIR").unwrap());

    let index = csr::load_index(base_dir.join(dataset), ratio).unwrap();
    let query_vectors = csr::load_vec(base_dir.join("queries.dev.csr")).unwrap();
    let mut it = query_vectors.iter().cycle();

    let pool = ScoresMemoryPool::new();
    let stopped = AtomicBool::new(false);

    c.bench_function(name, |b| {
        b.iter(|| {
            SearchContext::new(
                it.next().unwrap().clone(),
                TOP,
                &index,
                pool.get(),
                &stopped,
            )
            .search(&|_| true)
        })
    });
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = bench_search,
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_search,
}

criterion_main!(benches);
