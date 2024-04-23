use std::cmp::Ordering;
use std::io::{Read, Write};
use std::ops::ControlFlow;

use bitpacking::BitPacker as _;
use common::types::PointOffsetType;
#[cfg(debug_assertions)]
use itertools::Itertools as _;

use super::posting_list::PostingElement;
use crate::common::types::DimWeight;
type BitPackerImpl = bitpacking::BitPacker4x;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct PostingList {
    id_data: Vec<u8>,
    chunks: Vec<CompressedPostingChunk>,
    remainders: Vec<PostingElement0>,

    /// Copy of the last element in the list.
    /// Used to avoid unpacking the last chunk.
    last: Option<PostingElement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressedPostingChunk {
    /// Initial data point id. Used for decompression.
    initial: PointOffsetType,

    /// An offset within id_data
    offset: u32,

    /// Weight values for the chunk.
    weights: [DimWeight; BitPackerImpl::BLOCK_LEN],
}

impl PostingList {
    /// used for testing
    pub fn from(records: Vec<(PointOffsetType, DimWeight)>) -> PostingList {
        let mut posting_list = PostingBuilder::new();
        for (id, weight) in records {
            posting_list.add(id, weight);
        }
        posting_list.build()
    }

    /// Creates a new posting list with a single element.
    pub fn new_one(record_id: PointOffsetType, weight: DimWeight) -> PostingList {
        let mut builder = PostingBuilder::new();
        builder.add(record_id, weight);
        builder.build()
    }

    pub fn iter(&self) -> PostingListIterator {
        PostingListIterator::new(self)
    }

    pub fn len(&self) -> usize {
        self.chunks.len() * BitPackerImpl::BLOCK_LEN + self.remainders.len()
    }

    pub fn to_vec(&self) -> Vec<PostingElement> {
        // TODO: optimize
        self.iter().collect()
    }

    pub fn upsert(&mut self, element: PostingElement) {
        if self
            .last
            .as_ref()
            .map_or(true, |last| last.record_id < element.record_id)
        {
            self.last = Some(element.clone());
            self.remainders.push(PostingElement0 {
                record_id: element.record_id,
                weight: element.weight,
            });

            self.repack_remainders();
        } else {
            unimplemented!("Update is not implemented");
        }
    }

    fn repack_remainders(&mut self) {
        if self.remainders.len() < BitPackerImpl::BLOCK_LEN {
            return;
        }

        let chunk = CompressedPostingChunk {
            initial: self.remainders[0].record_id,
            offset: self.id_data.len() as u32,
            weights: self
                .remainders
                .iter()
                .map(|e| e.weight)
                .collect::<Vec<_>>()
                .try_into()
                .expect("Invalid chunk size"),
        };

        let mut this_chunk = [0u32; BitPackerImpl::BLOCK_LEN];
        for (i, e) in self.remainders.iter().enumerate() {
            this_chunk[i] = e.record_id;
        }

        let bitpacker = BitPackerImpl::new();
        let chunk_bits =
            bitpacker.num_bits_strictly_sorted(chunk.initial.checked_sub(1), &this_chunk);

        self.id_data.resize(
            self.id_data.len() + BitPackerImpl::compressed_block_size(chunk_bits),
            0,
        );
        bitpacker.compress_strictly_sorted(
            chunk.initial.checked_sub(1),
            &this_chunk,
            &mut self.id_data[chunk.offset as usize..],
            chunk_bits,
        );

        self.chunks.push(chunk);
        self.remainders.clear();
    }

    fn get_chunk_size(chunks: &[CompressedPostingChunk], data: &[u8], chunk_index: usize) -> usize {
        assert!(chunk_index < chunks.len());
        if chunk_index + 1 < chunks.len() {
            chunks[chunk_index + 1].offset as usize - chunks[chunk_index].offset as usize
        } else {
            data.len() - chunks[chunk_index].offset as usize
        }
    }

    fn decompress_chunk(
        &self,
        chunk_index: usize,
        decompressed_chunk: &mut [PointOffsetType; BitPackerImpl::BLOCK_LEN],
    ) {
        let chunk = &self.chunks[chunk_index];
        let chunk_size = Self::get_chunk_size(&self.chunks, &self.id_data, chunk_index);
        let chunk_bits = (chunk_size * 8) / BitPackerImpl::BLOCK_LEN;
        BitPackerImpl::new().decompress_strictly_sorted(
            chunk.initial.checked_sub(1),
            &self.id_data[chunk.offset as usize..chunk.offset as usize + chunk_size],
            decompressed_chunk,
            chunk_bits as u8,
        );
    }

    pub fn save(&self, file: &mut impl Write) -> std::io::Result<()> {
        file.write_all(&(self.id_data.len() as u32).to_ne_bytes())?;
        file.write_all(&(self.chunks.len() as u32).to_ne_bytes())?;
        file.write_all(&(self.remainders.len() as u32).to_ne_bytes())?;

        file.write_all(&self.id_data)?;
        for chunk in &self.chunks {
            file.write_all(&chunk.initial.to_ne_bytes())?;
            file.write_all(&chunk.offset.to_ne_bytes())?;
            for w in &chunk.weights {
                file.write_all(&w.to_ne_bytes())?;
            }
        }
        for e in &self.remainders {
            file.write_all(&e.record_id.to_ne_bytes())?;
            file.write_all(&e.weight.to_ne_bytes())?;
        }

        Ok(())
    }

    pub fn load(file: &mut impl Read) -> std::io::Result<PostingList> {
        let mut buf = [0u8; 12];
        file.read_exact(&mut buf)?;
        let id_data_len = u32::from_ne_bytes(buf[0..4].try_into().unwrap()) as usize;
        let chunks_len = u32::from_ne_bytes(buf[4..8].try_into().unwrap()) as usize;
        let remainders_len = u32::from_ne_bytes(buf[8..12].try_into().unwrap()) as usize;

        let mut id_data = vec![0u8; id_data_len];
        file.read_exact(&mut id_data)?;

        let mut chunks = Vec::with_capacity(chunks_len);
        for _ in 0..chunks_len {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            let initial = PointOffsetType::from_ne_bytes(buf);
            file.read_exact(&mut buf)?;
            let offset = u32::from_ne_bytes(buf);

            let mut weights = [0.0; BitPackerImpl::BLOCK_LEN];
            for w in &mut weights {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *w = f32::from_ne_bytes(buf);
            }
            chunks.push(CompressedPostingChunk {
                initial,
                offset,
                weights,
            });
        }

        let mut remainders = Vec::with_capacity(remainders_len);
        for _ in 0..remainders_len {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            let record_id = PointOffsetType::from_ne_bytes(buf);
            file.read_exact(&mut buf)?;
            let weight = f32::from_ne_bytes(buf);
            remainders.push(PostingElement0 { record_id, weight });
        }

        let last = if let Some(e) = remainders.last() {
            Some(PostingElement {
                record_id: e.record_id,
                weight: e.weight,
                max_next_weight: e.weight,
            })
        } else if let Some(chunk) = chunks.last() {
            let mut decompressed_chunk = [0; BitPackerImpl::BLOCK_LEN];
            let chunk_size = PostingList::get_chunk_size(&chunks, &id_data, chunks.len() - 1);
            BitPackerImpl::new().decompress_strictly_sorted(
                chunk.initial.checked_sub(1),
                &id_data[chunk.offset as usize..chunk.offset as usize + chunk_size],
                &mut decompressed_chunk,
                ((chunk_size * 8) / BitPackerImpl::BLOCK_LEN) as u8,
            );
            Some(PostingElement {
                record_id: decompressed_chunk[BitPackerImpl::BLOCK_LEN - 1],
                weight: chunk.weights[BitPackerImpl::BLOCK_LEN - 1],
                max_next_weight: chunk.weights[BitPackerImpl::BLOCK_LEN - 1],
            })
        } else {
            None
        };

        Ok(PostingList {
            id_data,
            chunks,
            remainders,
            last,
        })
    }
}

#[derive(Default)]
pub struct PostingBuilder {
    elements: Vec<PostingElement0>,
}

#[derive(Debug, Clone, PartialEq)]
struct PostingElement0 {
    record_id: PointOffsetType,
    weight: DimWeight,
}

impl PostingBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Add a new record to the posting list.
    pub fn add(&mut self, record_id: PointOffsetType, weight: DimWeight) {
        self.elements.push(PostingElement0 { record_id, weight });
    }

    pub fn build(mut self) -> PostingList {
        self.elements.sort_unstable_by_key(|e| e.record_id);

        // Check for duplicates
        #[cfg(debug_assertions)]
        if let Some(e) = self.elements.iter().duplicates_by(|e| e.record_id).next() {
            panic!("Duplicate id {} in posting list", e.record_id);
        }

        let mut this_chunk = Vec::with_capacity(BitPackerImpl::BLOCK_LEN);

        let bitpacker = BitPackerImpl::new();
        let mut chunks = Vec::with_capacity(self.elements.len() / BitPackerImpl::BLOCK_LEN);
        let mut data_size = 0;
        let mut remainders = Vec::new();
        for chunk in self.elements.chunks(BitPackerImpl::BLOCK_LEN) {
            if chunk.len() == BitPackerImpl::BLOCK_LEN {
                this_chunk.clear();
                this_chunk.extend(chunk.iter().map(|e| e.record_id));

                let initial = this_chunk[0];
                let chunk_bits =
                    bitpacker.num_bits_strictly_sorted(initial.checked_sub(1), &this_chunk);
                let chunk_size = BitPackerImpl::compressed_block_size(chunk_bits);
                chunks.push(CompressedPostingChunk {
                    initial,
                    offset: data_size as u32,
                    weights: chunk
                        .iter()
                        .map(|e| e.weight)
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("Invalid chunk size"),
                });
                data_size += chunk_size;
            } else {
                remainders.extend_from_slice(chunk);
            }
        }

        let mut id_data = vec![0u8; data_size];
        for (chunk_index, chunk_data) in self
            .elements
            .chunks_exact(BitPackerImpl::BLOCK_LEN)
            .enumerate()
        {
            this_chunk.clear();
            this_chunk.extend(chunk_data.iter().map(|e| e.record_id));

            let chunk = &chunks[chunk_index];
            let chunk_size = PostingList::get_chunk_size(&chunks, &id_data, chunk_index);
            let chunk_bits = (chunk_size * 8) / BitPackerImpl::BLOCK_LEN;
            bitpacker.compress_strictly_sorted(
                chunk.initial.checked_sub(1),
                &this_chunk,
                &mut id_data[chunk.offset as usize..chunk.offset as usize + chunk_size],
                chunk_bits as u8,
            );
        }

        PostingList {
            id_data,
            chunks,
            remainders,
            last: self.elements.last().map(|e| PostingElement {
                record_id: e.record_id,
                weight: e.weight,
                max_next_weight: e.weight,
            }),
        }
    }
}

/// Iterator over posting list elements offering skipping abilities to avoid full iteration.
#[derive(Clone)]
pub struct PostingListIterator<'a> {
    list: &'a PostingList,

    /// If true, then `decompressed_chunk` contains the unpacked chunk for the current
    /// `compressed_idx`.
    unpacked: bool,

    decompressed_chunk: [PointOffsetType; BitPackerImpl::BLOCK_LEN],

    /// Index within compressed chunks.
    compressed_idx: usize,

    remainders_idx: usize,
}

impl<'a> PostingListIterator<'a> {
    #[inline]
    fn new(list: &'a PostingList) -> PostingListIterator<'a> {
        PostingListIterator {
            list,
            unpacked: false,
            decompressed_chunk: [0; BitPackerImpl::BLOCK_LEN],
            compressed_idx: 0,
            remainders_idx: 0,
        }
    }

    #[inline]
    pub fn peek(&mut self) -> Option<PostingElement> {
        match self.try_for_each(ControlFlow::Break) {
            ControlFlow::Break(e) => Some(e),
            _ => None,
        }
    }

    #[inline]
    pub fn next(&mut self) -> Option<PostingElement> {
        let mut result = None;
        let mut first = true;
        self.try_for_each(|e| {
            if first {
                result = Some(e);
                first = false;
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        });
        result
    }

    #[inline]
    pub fn last(&self) -> Option<PostingElement> {
        self.list.last.clone()
    }

    pub fn len_to_end(&self) -> usize {
        self.list.len() - self.compressed_idx - self.remainders_idx
    }

    pub fn skip_to(&mut self, id: PointOffsetType) -> Option<PostingElement> {
        // TODO: optimize
        while let Some(e) = self.peek() {
            match e.record_id.cmp(&id) {
                Ordering::Equal => return Some(e),
                Ordering::Greater => return None,
                Ordering::Less => {
                    self.next();
                }
            }
        }
        None
    }

    pub fn skip_to_end(&mut self) {
        self.compressed_idx = self.list.chunks.len() * BitPackerImpl::BLOCK_LEN;
        self.remainders_idx = self.list.remainders.len();
    }

    pub fn try_for_each<F, R>(&mut self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(PostingElement) -> ControlFlow<R>,
    {
        let mut compressed_idx = self.compressed_idx;
        if compressed_idx / BitPackerImpl::BLOCK_LEN < self.list.chunks.len() {
            // 1. Iterate over already decompressed chunk
            if self.unpacked {
                let chunk = &self.list.chunks[compressed_idx / BitPackerImpl::BLOCK_LEN];

                for (idx, weight) in std::iter::zip(
                    &self.decompressed_chunk[compressed_idx % BitPackerImpl::BLOCK_LEN..],
                    &chunk.weights[compressed_idx % BitPackerImpl::BLOCK_LEN..],
                ) {
                    let res = f(PostingElement {
                        record_id: *idx,
                        weight: *weight,
                        max_next_weight: *weight, // TODO
                    });
                    if let ControlFlow::Break(_) = res {
                        self.compressed_idx = compressed_idx;
                        return res;
                    }
                    compressed_idx += 1;
                }
            }

            // 2. Iterate over compressed chunks
            while compressed_idx / BitPackerImpl::BLOCK_LEN < self.list.chunks.len() {
                self.list.decompress_chunk(
                    compressed_idx / BitPackerImpl::BLOCK_LEN,
                    &mut self.decompressed_chunk,
                );
                let chunk = &self.list.chunks[compressed_idx / BitPackerImpl::BLOCK_LEN];

                for (idx, weight) in std::iter::zip(&self.decompressed_chunk, &chunk.weights) {
                    let res = f(PostingElement {
                        record_id: *idx,
                        weight: *weight,
                        max_next_weight: *weight, // TODO
                    });
                    if let ControlFlow::Break(_) = res {
                        self.compressed_idx = compressed_idx;
                        self.unpacked = true;
                        return res;
                    }
                    compressed_idx += 1;
                }
            }
        }
        self.compressed_idx = compressed_idx;

        // 3. Iterate over remainders
        for e in &self.list.remainders[self.remainders_idx..] {
            f(PostingElement {
                record_id: e.record_id,
                weight: e.weight,
                max_next_weight: e.weight, // TODO
            })?;
            self.remainders_idx += 1;
        }

        ControlFlow::Continue(())
    }
}

impl Iterator for PostingListIterator<'_> {
    type Item = PostingElement;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: [usize; 6] = [0, 64, 128, 192, 256, 320];

    fn mk_case(count: usize) -> Vec<(PointOffsetType, DimWeight)> {
        (0..count)
            .map(|i| (i as u32 + 10000, i as DimWeight))
            .collect()
    }

    fn cases() -> Vec<Vec<(PointOffsetType, DimWeight)>> {
        CASES.iter().copied().map(mk_case).collect()
    }

    #[test]
    fn test_iter() {
        for case in cases() {
            let list = PostingList::from(case.clone());

            let mut iter = list.iter();

            let mut count = 0;

            assert_eq!(iter.len_to_end(), case.len(), "len_to_end");

            while let Some(e) = iter.next() {
                assert_eq!(e.record_id, case[count].0);
                assert_eq!(e.weight, case[count].1);
                assert_eq!(iter.len_to_end(), case.len() - count - 1);
                count += 1;
            }
        }
    }

    #[test]
    fn test_upsert_append() {
        for case in cases() {
            let mut pl = PostingList::default();
            for (id, weight) in case.iter().copied() {
                pl.upsert(PostingElement {
                    record_id: id,
                    weight,
                    max_next_weight: weight,
                });
                assert_eq!(pl.iter().last().unwrap().record_id, id);
            }
            let data = pl
                .iter()
                .map(|e| (e.record_id, e.weight))
                .collect::<Vec<_>>();
            assert_eq!(data, case);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // for consistency
    fn test_try_foreach() {
        for i in 0..CASES.len() {
            for j in i..CASES.len() {
                for k in j..CASES.len() {
                    eprintln!("\n\n\n{} {} {}", CASES[i], CASES[j], CASES[k]);
                    let case = mk_case(CASES[k]);
                    let pl = PostingList::from(case.clone());

                    let mut iter = pl.iter();

                    let mut data = Vec::new();
                    let mut counter = 0;
                    let end = iter.try_for_each(|e| {
                        eprintln!("  {}", e.record_id);
                        if counter == CASES[i] {
                            ControlFlow::Break(e.record_id)
                        } else {
                            data.push(e.record_id);
                            counter += 1;
                            ControlFlow::Continue(())
                        }
                    });
                    let end = match end {
                        ControlFlow::Continue(()) => None,
                        ControlFlow::Break(id) => Some(id),
                    };
                    assert_eq!(end, case.get(CASES[i]).map(|e| e.0));
                    assert_eq!(
                        data,
                        case[..CASES[i]].iter().map(|e| e.0).collect::<Vec<_>>()
                    );
                    eprintln!(" ;");

                    let mut data = Vec::new();
                    let mut counter = 0;
                    let end = iter.try_for_each(|e| {
                        eprintln!("  {}", e.record_id);
                        if counter == CASES[j] - CASES[i] {
                            ControlFlow::Break(e.record_id)
                        } else {
                            data.push(e.record_id);
                            counter += 1;
                            ControlFlow::Continue(())
                        }
                    });
                    let end = match end {
                        ControlFlow::Continue(()) => None,
                        ControlFlow::Break(id) => Some(id),
                    };
                    assert_eq!(end, case.get(CASES[j]).map(|e| e.0));
                    assert_eq!(
                        data,
                        if i != j {
                            case[CASES[i]..CASES[j]]
                                .iter()
                                .map(|e| e.0)
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn test_load_and_save() {
        for case in cases() {
            let list = PostingList::from(case.clone());

            let mut buf = Vec::new();
            list.save(&mut buf).unwrap();

            let mut file = std::io::Cursor::new(buf);
            let list2 = PostingList::load(&mut file).unwrap();

            let data = list2
                .iter()
                .map(|e| (e.record_id, e.weight))
                .collect::<Vec<_>>();
            assert_eq!(data, case);

            assert_eq!(list.iter().last(), list2.iter().last());
        }
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    const DEFAULT_MAX_NEXT_WEIGHT: DimWeight = f32::NEG_INFINITY; // ???

    #[test]
    fn test_posting_operations() {
        let mut builder = PostingBuilder::new();
        builder.add(1, 1.0);
        builder.add(2, 2.1);
        builder.add(5, 5.0);
        builder.add(3, 2.0);
        builder.add(8, 3.4);
        builder.add(10, 3.0);
        builder.add(20, 3.0);
        builder.add(7, 4.0);
        builder.add(11, 3.0);

        let posting_list = builder.build();

        let mut iter = posting_list.iter();

        assert_eq!(iter.peek().unwrap().record_id, 1);
        iter.next();
        assert_eq!(iter.peek().unwrap().record_id, 2);
        iter.next();
        assert_eq!(iter.peek().unwrap().record_id, 3);

        assert_eq!(iter.skip_to(7).unwrap().record_id, 7);
        assert_eq!(iter.peek().unwrap().record_id, 7);

        assert!(iter.skip_to(9).is_none());
        assert_eq!(iter.peek().unwrap().record_id, 10);

        assert!(iter.skip_to(20).is_some());
        assert_eq!(iter.peek().unwrap().record_id, 20);

        assert!(iter.skip_to(21).is_none());
        assert!(iter.peek().is_none());
    }

    #[test]
    fn test_upsert_insert_last() {
        let mut builder = PostingBuilder::new();
        builder.add(1, 1.0);
        builder.add(3, 3.0);
        builder.add(2, 2.0);

        let mut posting_list = builder.build();

        let vec = posting_list.to_vec();

        assert_eq!(
            vec,
            vec![
                PostingElement {
                    record_id: 1,
                    weight: 1.0,
                    max_next_weight: 1.0
                },
                PostingElement {
                    record_id: 2,
                    weight: 2.0,
                    max_next_weight: 2.0
                },
                PostingElement {
                    record_id: 3,
                    weight: 3.0,
                    max_next_weight: 3.0
                }
            ]
        );

        // sorted by id
        /*
        assert_eq!(posting_list.elements[0].record_id, 1);
        assert_eq!(posting_list.elements[0].weight, 1.0);
        assert_eq!(posting_list.elements[0].max_next_weight, 3.0);

        assert_eq!(posting_list.elements[1].record_id, 2);
        assert_eq!(posting_list.elements[1].weight, 2.0);
        assert_eq!(posting_list.elements[1].max_next_weight, 3.0);

        assert_eq!(posting_list.elements[2].record_id, 3);
        assert_eq!(posting_list.elements[2].weight, 3.0);
        assert_eq!(
            posting_list.elements[2].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );

        // insert mew last element
        posting_list.upsert(PostingElement::new(4, 4.0));
        assert_eq!(posting_list.elements[3].record_id, 4);
        assert_eq!(posting_list.elements[3].weight, 4.0);
        assert_eq!(
            posting_list.elements[3].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );

        // must update max_next_weight of previous elements if necessary
        for element in posting_list.elements.iter().take(3) {
            assert_eq!(element.max_next_weight, 4.0);
        }
        */
    }

    /*
    #[test]
    fn test_upsert_insert_in_gap() {
        let mut builder = PostingBuilder::new();
        builder.add(1, 1.0);
        builder.add(3, 3.0);
        builder.add(2, 2.0);
        // no entry for 4
        builder.add(5, 5.0);

        let mut posting_list = builder.build();

        // sorted by id
        assert_eq!(posting_list.elements[0].record_id, 1);
        assert_eq!(posting_list.elements[0].weight, 1.0);
        assert_eq!(posting_list.elements[0].max_next_weight, 5.0);

        assert_eq!(posting_list.elements[1].record_id, 2);
        assert_eq!(posting_list.elements[1].weight, 2.0);
        assert_eq!(posting_list.elements[1].max_next_weight, 5.0);

        assert_eq!(posting_list.elements[2].record_id, 3);
        assert_eq!(posting_list.elements[2].weight, 3.0);
        assert_eq!(posting_list.elements[2].max_next_weight, 5.0);

        assert_eq!(posting_list.elements[3].record_id, 5);
        assert_eq!(posting_list.elements[3].weight, 5.0);
        assert_eq!(
            posting_list.elements[3].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );

        // insert mew last element
        posting_list.upsert(PostingElement::new(4, 4.0));

        // `4` is shifted to the right
        assert_eq!(posting_list.elements[4].record_id, 5);
        assert_eq!(posting_list.elements[4].weight, 5.0);
        assert_eq!(
            posting_list.elements[4].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );

        // new element
        assert_eq!(posting_list.elements[3].record_id, 4);
        assert_eq!(posting_list.elements[3].weight, 4.0);

        // must update max_next_weight of previous elements
        for element in posting_list.elements.iter().take(4) {
            assert_eq!(element.max_next_weight, 5.0);
        }
    }

    #[test]
    fn test_upsert_update() {
        let mut builder = PostingBuilder::new();
        builder.add(1, 1.0);
        builder.add(3, 3.0);
        builder.add(2, 2.0);

        let mut posting_list = builder.build();

        // sorted by id
        assert_eq!(posting_list.elements[0].record_id, 1);
        assert_eq!(posting_list.elements[0].weight, 1.0);
        assert_eq!(posting_list.elements[0].max_next_weight, 3.0);

        assert_eq!(posting_list.elements[1].record_id, 2);
        assert_eq!(posting_list.elements[1].weight, 2.0);
        assert_eq!(posting_list.elements[1].max_next_weight, 3.0);

        assert_eq!(posting_list.elements[2].record_id, 3);
        assert_eq!(posting_list.elements[2].weight, 3.0);
        assert_eq!(
            posting_list.elements[2].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );

        // increase weight of existing element
        posting_list.upsert(PostingElement::new(2, 4.0));

        assert_eq!(posting_list.elements[0].record_id, 1);
        assert_eq!(posting_list.elements[0].weight, 1.0);
        assert_eq!(posting_list.elements[0].max_next_weight, 4.0); // update propagated

        assert_eq!(posting_list.elements[1].record_id, 2);
        assert_eq!(posting_list.elements[1].weight, 4.0); // updated
        assert_eq!(posting_list.elements[1].max_next_weight, 3.0);

        assert_eq!(posting_list.elements[2].record_id, 3);
        assert_eq!(posting_list.elements[2].weight, 3.0);
        assert_eq!(
            posting_list.elements[2].max_next_weight,
            DEFAULT_MAX_NEXT_WEIGHT
        );
    }
    */
}
