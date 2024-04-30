use std::cmp::max;
use std::fs::File;
use std::io::{Read as _, Seek as _};
use std::path::{Path, PathBuf};

use common::types::PointOffsetType;
use io::file_operations::{atomic_save_json, read_json};

use super::inverted_index_mmap::{InvertedIndexFileHeader, InvertedIndexMmap};
use crate::common::sparse_vector::RemappedSparseVector;
use crate::common::types::DimId;
use crate::index::inverted_index::inverted_index_mmap::PostingListFileHeader;
use crate::index::inverted_index::InvertedIndex;
use crate::index::posting_list::PostingElement;
use crate::index::posting_list2::{PostingList, PostingListIterator};

/// Inverted flatten index from dimension id to posting list
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedIndexRam {
    /// Posting lists for each dimension flattened (dimension id -> posting list)
    /// Gaps are filled with empty posting lists
    pub postings: Vec<PostingList>,
    /// Number of unique indexed vectors
    /// pre-computed on build and upsert to avoid having to traverse the posting lists.
    pub vector_count: usize,
}

impl InvertedIndex for InvertedIndexRam {
    fn open(path: &Path) -> std::io::Result<Self> {
        let mut postings = Vec::new();

        // read index config file
        let config_file_path = InvertedIndexMmap::index_config_file_path(path);
        // if the file header does not exist, the index is malformed
        let file_header: InvertedIndexFileHeader = read_json(&config_file_path)?;
        // read index data into mmap

        let file_path = InvertedIndexMmap::index_file_path(path);
        match file_header.version {
            0 => {
                log::info!("Converting old sparse index format for {}", path.display());

                let mut file = File::open(&file_path)?;

                // Read headers
                let mut headers = Vec::with_capacity(
                    file_header.posting_count * std::mem::size_of::<PostingListFileHeader>(),
                );
                file.by_ref()
                    .take(
                        (file_header.posting_count * std::mem::size_of::<PostingListFileHeader>())
                            as u64,
                    )
                    .read_to_end(&mut headers)?;
                if headers.len()
                    != file_header.posting_count * std::mem::size_of::<PostingListFileHeader>()
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to read all headers {}", headers.len()),
                    ));
                }
                let headers: Vec<PostingListFileHeader> =
                    transmute_vec(headers).expect("Failed to transmute headers");

                let mut postings = Vec::with_capacity(file_header.posting_count);
                let mut buf = Vec::<u8>::new();
                for header in headers {
                    file.seek(std::io::SeekFrom::Start(header.start_offset))?;
                    buf.resize((header.end_offset - header.start_offset) as usize, 0);
                    file.read_exact(&mut buf)?;

                    let (head, body, tail) = unsafe { buf.align_to::<PostingElement>() };
                    if !head.is_empty() || !tail.is_empty() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Failed to align to PostingElement",
                        ));
                    }

                    let mut posting_list = PostingList::default();
                    for element in body.iter().cloned() {
                        posting_list.upsert(element);
                    }
                    postings.push(posting_list);
                }

                let index = InvertedIndexRam {
                    postings,
                    vector_count: file_header.vector_count,
                };

                index.save(path)?;
                Ok(index)
            }
            // Current version
            1 => {
                let file = File::open(&file_path)?;
                let mut bufreader = std::io::BufReader::new(&file);
                for _ in 0..file_header.posting_count {
                    postings.push(PostingList::load(&mut bufreader)?);
                }

                Ok(InvertedIndexRam {
                    postings,
                    vector_count: file_header.vector_count,
                })
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", file_header.version),
            )),
        }
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        // InvertedIndexMmap::convert_and_save(self, path)?;

        let file_path = InvertedIndexMmap::index_file_path(path);
        let mut file = File::create(file_path)?;
        for posting in &self.postings {
            posting.save(&mut file)?;
        }

        // save header properties
        let posting_count = self.postings.len();
        let vector_count = self.vector_count();

        // finalize data with index file.
        let file_header = InvertedIndexFileHeader {
            posting_count,
            vector_count,
            version: 1,
        };

        let config_file_path = InvertedIndexMmap::index_config_file_path(path);
        atomic_save_json(&config_file_path, &file_header)?;

        Ok(())
    }

    fn get(&self, id: &DimId) -> Option<PostingListIterator> {
        self.get(id).map(|posting_list| posting_list.iter())
    }

    fn len(&self) -> usize {
        self.postings.len()
    }

    fn posting_list_len(&self, id: &DimId) -> Option<usize> {
        self.get(id).map(|posting_list| posting_list.len())
    }

    fn files(path: &Path) -> Vec<PathBuf> {
        [
            InvertedIndexMmap::index_file_path(path),
            InvertedIndexMmap::index_config_file_path(path),
        ]
        .into_iter()
        .filter(|p| p.exists())
        .collect()
    }

    fn upsert(&mut self, id: PointOffsetType, vector: RemappedSparseVector) {
        self.upsert(id, vector);
    }

    fn from_ram_index<P: AsRef<Path>>(
        ram_index: InvertedIndexRam,
        _path: P,
    ) -> std::io::Result<Self> {
        Ok(ram_index)
    }

    fn vector_count(&self) -> usize {
        self.vector_count
    }

    fn max_index(&self) -> Option<DimId> {
        match self.postings.len() {
            0 => None,
            len => Some(len as DimId - 1),
        }
    }
}

impl InvertedIndexRam {
    /// New empty inverted index
    pub fn empty() -> InvertedIndexRam {
        InvertedIndexRam {
            postings: Vec::new(),
            vector_count: 0,
        }
    }

    /// Get posting list for dimension id
    pub fn get(&self, id: &DimId) -> Option<&PostingList> {
        self.postings.get((*id) as usize)
    }

    /// Upsert a vector into the inverted index.
    pub fn upsert(&mut self, id: PointOffsetType, vector: RemappedSparseVector) {
        for (dim_id, weight) in vector.indices.into_iter().zip(vector.values.into_iter()) {
            let dim_id = dim_id as usize;
            match self.postings.get_mut(dim_id) {
                Some(posting) => {
                    // update existing posting list
                    let posting_element = PostingElement::new(id, weight);
                    posting.upsert(posting_element);
                }
                None => {
                    // resize postings vector (fill gaps with empty posting lists)
                    self.postings.resize_with(dim_id + 1, PostingList::default);
                    // initialize new posting for dimension
                    self.postings[dim_id] = PostingList::new_one(id, weight);
                }
            }
        }
        // given that there are no holes in the internal ids and that we are not deleting from the index
        // we can just use the id as a proxy the count
        self.vector_count = max(self.vector_count, id as usize);
    }
}

fn transmute_vec<T>(mut v: Vec<u8>) -> Result<Vec<T>, std::io::Error> {
    let len = v.len();
    let ptr = v.as_mut_ptr();

    if len % std::mem::size_of::<T>() != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid size",
        ));
    }
    if ptr.align_offset(std::mem::align_of::<T>()) != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid alignment",
        ));
    }

    std::mem::forget(v);
    Ok(unsafe {
        Vec::from_raw_parts(
            ptr as *mut T,
            len / std::mem::size_of::<T>(),
            len / std::mem::size_of::<T>(),
        )
    })
}

#[cfg(test)]
mod tests {
    use tempfile::Builder;

    use super::*;
    use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

    #[test]
    fn upsert_same_dimension_inverted_index_ram() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, [(1, 10.0), (2, 10.0), (3, 10.0)].into());
        builder.add(2, [(1, 20.0), (2, 20.0), (3, 20.0)].into());
        builder.add(3, [(1, 30.0), (2, 30.0), (3, 30.0)].into());
        let mut inverted_index_ram = builder.build();

        assert_eq!(inverted_index_ram.vector_count, 3);

        inverted_index_ram.upsert(
            4,
            RemappedSparseVector::new(vec![1, 2, 3], vec![40.0, 40.0, 40.0]).unwrap(),
        );
        for i in 1..4 {
            let posting_list = inverted_index_ram.get(&i).unwrap();
            let posting_list = posting_list.to_vec();
            assert_eq!(posting_list.len(), 4);
            assert_eq!(posting_list.first().unwrap().weight, 10.0);
            assert_eq!(posting_list.get(1).unwrap().weight, 20.0);
            assert_eq!(posting_list.get(2).unwrap().weight, 30.0);
            assert_eq!(posting_list.get(3).unwrap().weight, 40.0);
        }
    }

    #[test]
    fn upsert_new_dimension_inverted_index_ram() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, [(1, 10.0), (2, 10.0), (3, 10.0)].into());
        builder.add(2, [(1, 20.0), (2, 20.0), (3, 20.0)].into());
        builder.add(3, [(1, 30.0), (2, 30.0), (3, 30.0)].into());
        let mut inverted_index_ram = builder.build();

        assert_eq!(inverted_index_ram.vector_count, 3);

        // 4 postings, 0th empty
        assert_eq!(inverted_index_ram.postings.len(), 4);

        inverted_index_ram.upsert(
            4,
            RemappedSparseVector::new(vec![1, 2, 30], vec![40.0, 40.0, 40.0]).unwrap(),
        );

        // new dimension resized postings
        assert_eq!(inverted_index_ram.postings.len(), 31);

        // updated existing dimension
        for i in 1..3 {
            let posting_list = inverted_index_ram.get(&i).unwrap();
            let posting_list = posting_list.to_vec();
            assert_eq!(posting_list.len(), 4);
            assert_eq!(posting_list.first().unwrap().weight, 10.0);
            assert_eq!(posting_list.get(1).unwrap().weight, 20.0);
            assert_eq!(posting_list.get(2).unwrap().weight, 30.0);
            assert_eq!(posting_list.get(3).unwrap().weight, 40.0);
        }

        // fetch 30th posting
        let postings = inverted_index_ram.get(&30).unwrap();
        let postings = postings.to_vec();
        assert_eq!(postings.len(), 1);
        let posting = postings.first().unwrap();
        assert_eq!(posting.record_id, 4);
        assert_eq!(posting.weight, 40.0);
    }

    #[test]
    fn test_upsert_insert_equivalence() {
        let first_vec: RemappedSparseVector = [(1, 10.0), (2, 10.0), (3, 10.0)].into();
        let second_vec: RemappedSparseVector = [(1, 20.0), (2, 20.0), (3, 20.0)].into();
        let third_vec: RemappedSparseVector = [(1, 30.0), (2, 30.0), (3, 30.0)].into();

        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, first_vec.clone());
        builder.add(2, second_vec.clone());
        builder.add(3, third_vec.clone());
        let inverted_index_ram_built = builder.build();

        assert_eq!(inverted_index_ram_built.vector_count, 3);

        let mut inverted_index_ram_upserted = InvertedIndexRam::empty();
        inverted_index_ram_upserted.upsert(1, first_vec);
        inverted_index_ram_upserted.upsert(2, second_vec);
        inverted_index_ram_upserted.upsert(3, third_vec);

        assert_eq!(
            inverted_index_ram_built.postings.len(),
            inverted_index_ram_upserted.postings.len()
        );
        assert_eq!(inverted_index_ram_built, inverted_index_ram_upserted);
    }

    #[test]
    fn inverted_index_ram_save_load() {
        let mut builder = InvertedIndexBuilder::new();
        builder.add(1, vec![(1, 10.0), (2, 10.0), (3, 10.0)].try_into().unwrap());
        builder.add(2, vec![(1, 20.0), (2, 20.0), (3, 20.0)].try_into().unwrap());
        builder.add(3, vec![(1, 30.0), (2, 30.0), (3, 30.0)].try_into().unwrap());
        let inverted_index_ram = builder.build();

        assert_eq!(inverted_index_ram.vector_count, 3);

        let tmp_dir_path = Builder::new().prefix("test_index_dir").tempdir().unwrap();
        inverted_index_ram.save(tmp_dir_path.path()).unwrap();

        let loaded_inverted_index_ram = InvertedIndexRam::open(tmp_dir_path.path()).unwrap();
        assert_eq!(inverted_index_ram, loaded_inverted_index_ram);
    }
}
