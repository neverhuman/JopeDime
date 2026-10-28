//! Streaming, type-tagged identity encoding. Float payloads are their native bits;
//! no intermediate JSON can collapse NaN or infinity into null.
use serde::ser::{self, Serialize, Serializer};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, serde_json::Error>;
pub(super) struct Encoder<'a>(pub &'a mut Sha256);
impl Encoder<'_> {
    fn part(&mut self, tag: u8, bytes: &[u8]) {
        self.0.update([tag]);
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }
    fn begin(&mut self, tag: u8, len: Option<usize>) {
        self.part(tag, &len.map_or(u64::MAX, |n| n as u64).to_le_bytes());
    }
    fn end(&mut self) {
        self.part(b'e', &[]);
    }
}
macro_rules! integer {
    ($method:ident,$ty:ty,$tag:expr) => {
        fn $method(self, value: $ty) -> Result<()> {
            self.part($tag, &value.to_le_bytes());
            Ok(())
        }
    };
}
impl Serializer for &mut Encoder<'_> {
    type Ok = ();
    type Error = serde_json::Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;
    fn serialize_bool(self, v: bool) -> Result<()> {
        self.part(b'b', &[u8::from(v)]);
        Ok(())
    }
    integer!(serialize_i8, i8, b'1');
    integer!(serialize_i16, i16, b'2');
    integer!(serialize_i32, i32, b'3');
    integer!(serialize_i64, i64, b'4');
    integer!(serialize_i128, i128, b'5');
    integer!(serialize_u8, u8, b'6');
    integer!(serialize_u16, u16, b'7');
    integer!(serialize_u32, u32, b'8');
    integer!(serialize_u64, u64, b'9');
    integer!(serialize_u128, u128, b'0');
    fn serialize_f32(self, v: f32) -> Result<()> {
        self.part(b'f', &v.to_bits().to_le_bytes());
        Ok(())
    }
    fn serialize_f64(self, v: f64) -> Result<()> {
        self.part(b'd', &v.to_bits().to_le_bytes());
        Ok(())
    }
    fn serialize_char(self, v: char) -> Result<()> {
        let mut b = [0; 4];
        self.part(b'c', v.encode_utf8(&mut b).as_bytes());
        Ok(())
    }
    fn serialize_str(self, v: &str) -> Result<()> {
        self.part(b's', v.as_bytes());
        Ok(())
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        self.part(b'y', v);
        Ok(())
    }
    fn serialize_none(self) -> Result<()> {
        self.part(b'n', &[]);
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<()> {
        self.part(b'o', &[]);
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<()> {
        self.part(b'u', &[]);
        Ok(())
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<()> {
        self.part(b'U', name.as_bytes());
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.part(b'V', name.as_bytes());
        self.part(b'i', &index.to_le_bytes());
        self.part(b'v', variant.as_bytes());
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        v: &T,
    ) -> Result<()> {
        self.part(b'N', name.as_bytes());
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<()> {
        self.serialize_unit_variant(name, index, variant)?;
        self.part(b'J', &[]);
        v.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self> {
        self.begin(b'q', len);
        Ok(self)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self> {
        self.begin(b't', Some(len));
        Ok(self)
    }
    fn serialize_tuple_struct(self, name: &'static str, len: usize) -> Result<Self> {
        self.part(b'T', name.as_bytes());
        self.begin(b't', Some(len));
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self> {
        self.serialize_unit_variant(name, index, variant)?;
        self.begin(b'Q', Some(len));
        Ok(self)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self> {
        self.begin(b'm', len);
        Ok(self)
    }
    fn serialize_struct(self, name: &'static str, len: usize) -> Result<Self> {
        self.part(b'S', name.as_bytes());
        self.begin(b'r', Some(len));
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self> {
        self.serialize_unit_variant(name, index, variant)?;
        self.begin(b'R', Some(len));
        Ok(self)
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}
macro_rules! sequence {
    ($trait:ident,$method:ident) => {
        impl ser::$trait for &mut Encoder<'_> {
            type Ok = ();
            type Error = serde_json::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<()> {
                Encoder::end(self);
                Ok(())
            }
        }
    };
}
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);
impl ser::SerializeMap for &mut Encoder<'_> {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        v.serialize(&mut **self)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        v.serialize(&mut **self)
    }
    fn end(self) -> Result<()> {
        Encoder::end(self);
        Ok(())
    }
}
macro_rules! structure {
    ($trait:ident) => {
        impl ser::$trait for &mut Encoder<'_> {
            type Ok = ();
            type Error = serde_json::Error;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                v: &T,
            ) -> Result<()> {
                self.part(b'k', key.as_bytes());
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<()> {
                Encoder::end(self);
                Ok(())
            }
        }
    };
}
structure!(SerializeStruct);
structure!(SerializeStructVariant);
