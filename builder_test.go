package tdigest

import "testing"

func TestAllocBufs(t *testing.T) {
	b1, b2 := alloc2Bufs(10, 20)
	if len(b1) != 10 || cap(b1) != 10 {
		t.Errorf("bad b1: len=%d cap=%d", len(b1), cap(b1))
	}
	if len(b2) != 20 || cap(b2) != 20 {
		t.Errorf("bad b2: len=%d cap=%d", len(b2), cap(b2))
	}

	b1, b2, b3 := alloc3Bufs(10, 20, 30)
	if len(b1) != 10 || cap(b1) != 10 {
		t.Errorf("bad b1: len=%d cap=%d", len(b1), cap(b1))
	}
	if len(b2) != 20 || cap(b2) != 20 {
		t.Errorf("bad b2: len=%d cap=%d", len(b2), cap(b2))
	}
	if len(b3) != 30 || cap(b3) != 30 {
		t.Errorf("bad b3: len=%d cap=%d", len(b3), cap(b3))
	}
}
