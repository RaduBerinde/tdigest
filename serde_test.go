package tdigest

import (
	"encoding/binary"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestSerdeRandomized(t *testing.T) {
	var d2 TDigest
	for range 1000 {
		delta := 1 + rand.IntN(1000)
		bld := MakeBuilder(delta)
		for range 1 + rand.IntN(10000) {
			x := rand.NormFloat64()
			bld.Add(x, 1.0)
		}
		d := bld.Digest()
		n := d.SerializedSize()
		buf := d.Serialize(nil)
		if len(buf) != n {
			t.Fatalf("serialized size mismatch: SerializedSize()=%d, Serialized() returns %d bytes", n, len(buf))
		}
		if rand.IntN(2) == 0 {
			buf = append(buf, []byte("lolomg")...)
		}
		n2, err := Deserialize(&d2, buf)
		if err != nil {
			t.Fatalf("Deserialize() error: %v", err)
		}
		if n2 != n {
			t.Fatalf("Deserialize() returned offset %d, expecting %d", n2, n)
		}
		if !reflect.DeepEqual(d, d2) {
			t.Fatalf("tdigest mismatch after serialize/deserialize")
		}
	}
}

func TestSerdeEdgeCases(t *testing.T) {
	var d2 TDigest

	t.Run("empty digest", func(t *testing.T) {
		d := TDigest{delta: 100}
		buf := d.Serialize(nil)
		_, err := Deserialize(&d2, buf)
		if err != nil {
			t.Fatalf("Deserialize() error: %v", err)
		}
		if !reflect.DeepEqual(d, d2) {
			t.Fatalf("tdigest mismatch after serialize/deserialize")
		}
	})

	t.Run("single centroid", func(t *testing.T) {
		bld := MakeBuilder(100)
		bld.Add(42.0, 1.0)
		d := bld.Digest()
		buf := d.Serialize(nil)
		_, err := Deserialize(&d2, buf)
		if err != nil {
			t.Fatalf("Deserialize() error: %v", err)
		}
		if !reflect.DeepEqual(d, d2) {
			t.Fatalf("tdigest mismatch after serialize/deserialize")
		}
	})

	t.Run("two centroids", func(t *testing.T) {
		bld := MakeBuilder(100)
		bld.Add(10.0, 1.0)
		bld.Add(20.0, 1.0)
		d := bld.Digest()
		buf := d.Serialize(nil)
		_, err := Deserialize(&d2, buf)
		if err != nil {
			t.Fatalf("Deserialize() error: %v", err)
		}
		if !reflect.DeepEqual(d, d2) {
			t.Fatalf("tdigest mismatch after serialize/deserialize")
		}
	})
}

func TestSerdeErrors(t *testing.T) {
	// Create a valid digest for testing
	bld := MakeBuilder(100)
	bld.Add(1.0, 1.0)
	bld.Add(2.0, 1.0)
	d := bld.Digest()
	validBuf := d.Serialize(nil)

	expectErr := func(t *testing.T, buf []byte) {
		var d2 TDigest
		if _, err := Deserialize(&d2, buf); err == nil {
			t.Fatal("expected error")
		}
	}

	t.Run("buffer too short - header", func(t *testing.T) {
		expectErr(t, validBuf[:10])
	})

	t.Run("buffer too short - data", func(t *testing.T) {
		expectErr(t, validBuf[:len(validBuf)-5])
	})

	t.Run("invalid magic", func(t *testing.T) {
		buf := append([]byte{}, validBuf...)
		binary.LittleEndian.PutUint32(buf[0:], 0xDEADBEEF)
		expectErr(t, buf)
	})

	t.Run("invalid version", func(t *testing.T) {
		buf := append([]byte{}, validBuf...)
		binary.LittleEndian.PutUint32(buf[4:], 999)
		expectErr(t, buf)
	})
}

func TestSerdeAppendToBuffer(t *testing.T) {
	bld := MakeBuilder(100)
	for i := range 100 {
		bld.Add(float64(i), 1.0)
	}
	d := bld.Digest()

	// Create a non-empty buffer
	prefix := []byte("prefix data")
	buf := append([]byte{}, prefix...)

	// Serialize appending to the buffer
	buf = d.Serialize(buf)

	// Verify prefix is intact
	if string(buf[:len(prefix)]) != string(prefix) {
		t.Fatal("prefix data was corrupted")
	}

	// Deserialize from the offset
	var d2 TDigest
	offset, err := Deserialize(&d2, buf[len(prefix):])
	if err != nil {
		t.Fatalf("Deserialize() error: %v", err)
	}

	if offset != d.SerializedSize() {
		t.Fatalf("unexpected offset: got %d, want %d", offset, d.SerializedSize())
	}

	if !reflect.DeepEqual(d, d2) {
		t.Fatal("tdigest mismatch after serialize/deserialize")
	}
}
