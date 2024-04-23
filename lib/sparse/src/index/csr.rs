use std::io;
use std::path::Path;

use indicatif::ProgressBar;
use memmap2::Mmap;
use memory::mmap_ops::{open_read_mmap, transmute_from_u8, transmute_from_u8_to_slice};
use validator::ValidationErrors;

use super::inverted_index::inverted_index_ram::InvertedIndexRam;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

struct Csr<'a> {
    nrow: usize,
    indptr: &'a [u64],
    indices: &'a [u32],
    data: &'a [f32],
}

impl<'a> Csr<'a> {
    fn from_mmap(mmap: &'a Mmap) -> io::Result<Csr<'a>> {
        let mut pos = 0;
        let (nrow, ncol, nnz) = transmute_from_u8::<(u64, u64, u64)>(&mmap.as_ref()[pos..pos + 24]);
        let (nrow, ncol, nnz) = (*nrow as usize, *ncol as usize, *nnz as usize);
        pos += 24;

        let indptr = transmute_from_u8_to_slice::<u64>(&mmap.as_ref()[pos..pos + 8 * (nrow + 1)]);
        assert!(indptr.windows(2).all(|w| w[0] <= w[1]));
        pos += 8 * (nrow + 1);

        let indices = transmute_from_u8_to_slice::<u32>(&mmap.as_ref()[pos..pos + 4 * nnz]);
        assert!(indices.iter().all(|&i| i < ncol as u32));
        pos += 4 * nnz;

        let data = transmute_from_u8_to_slice::<f32>(&mmap.as_ref()[pos..pos + 4 * nnz]);

        Ok(Csr {
            nrow,
            indptr,
            indices,
            data,
        })
    }

    fn vec(&self, row: usize) -> Result<RemappedSparseVector, ValidationErrors> {
        RemappedSparseVector::new(
            self.indices[self.indptr[row] as usize..self.indptr[row + 1] as usize].to_vec(),
            self.data[self.indptr[row] as usize..self.indptr[row + 1] as usize].to_vec(),
        )
    }
}

pub fn load_index(path: impl AsRef<Path>, ratio: f32) -> io::Result<InvertedIndexRam> {
    let mmap = open_read_mmap(path.as_ref())?;
    let csr = Csr::from_mmap(&mmap).unwrap();
    let mut builder = InvertedIndexBuilder::new();

    assert!(ratio > 0.0 && ratio <= 1.0);
    let nrow = (csr.nrow as f32 * ratio) as usize;

    let bar = ProgressBar::new(nrow as u64);
    for row in 0..nrow {
        bar.inc(1);
        builder.add(row as u32, csr.vec(row).unwrap());
    }
    Ok(builder.build())
}

pub fn load_vec(path: impl AsRef<Path>) -> io::Result<Vec<RemappedSparseVector>> {
    let mmap = open_read_mmap(path.as_ref())?;
    let csr = Csr::from_mmap(&mmap).unwrap();
    let csr = (0..csr.nrow)
        .map(|i| csr.vec(i))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    Ok(csr)
}
