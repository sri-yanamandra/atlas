use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use parquet::errors::Result;
use parquet::file::reader::{ChunkReader, Length};

/// A zero-copy Parquet reader for a byte range inside a `.cohort` file.
///
/// Instead of reading the entire Parquet block into memory, this struct
/// implements the parquet crate's `ChunkReader` trait so the decoder can
/// seek directly to individual column chunks on disk.
///
/// Each `get_read()`/`get_bytes()` call opens a fresh `File` handle and
/// seeks to `start_offset + start`, the same pattern the parquet crate's
/// own `File` impl uses with `try_clone()`.
pub struct OffsetReader {
    path: PathBuf,
    start_offset: u64,
    length: u64,
}

impl OffsetReader {
    pub fn new(path: impl AsRef<Path>, start_offset: u64, length: u64) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            start_offset,
            length,
        }
    }
}

impl Length for OffsetReader {
    fn len(&self) -> u64 {
        self.length
    }
}

impl ChunkReader for OffsetReader {
    type T = BufReader<File>;

    fn get_read(&self, start: u64) -> Result<Self::T> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.start_offset + start))?;
        Ok(BufReader::new(file))
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.start_offset + start))?;
        let mut buf = vec![0u8; length];
        file.read_exact(&mut buf)?;
        Ok(Bytes::from(buf))
    }
}

/// A Read+Seek adapter for a byte range inside a `.cohort` file.
///
/// Arrow IPC's `FileReader` requires `Read + Seek` (it does `SeekFrom::End(-8)`
/// to locate the footer). This struct maps all seek/read operations to the
/// sub-range `[start_offset, start_offset + length)` inside the host file.
pub struct OffsetReadSeek {
    file: BufReader<File>,
    start_offset: u64,
    length: u64,
    position: u64, // relative to start_offset
}

impl OffsetReadSeek {
    pub fn new(path: impl AsRef<Path>, start_offset: u64, length: u64) -> io::Result<Self> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(start_offset))?;
        Ok(Self {
            file: BufReader::new(file),
            start_offset,
            length,
            position: 0,
        })
    }
}

impl Read for OffsetReadSeek {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.length.saturating_sub(self.position) as usize;
        if remaining == 0 {
            return Ok(0);
        }
        let to_read = buf.len().min(remaining);
        let n = self.file.read(&mut buf[..to_read])?;
        self.position += n as u64;
        Ok(n)
    }
}

impl Seek for OffsetReadSeek {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::Current(offset) => self.position as i64 + offset,
            SeekFrom::End(offset) => self.length as i64 + offset,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start of block",
            ));
        }
        let new_pos = (new_pos as u64).min(self.length);
        self.file
            .seek(SeekFrom::Start(self.start_offset + new_pos))?;
        self.position = new_pos;
        Ok(new_pos)
    }
}
