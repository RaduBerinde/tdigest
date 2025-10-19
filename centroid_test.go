package tdigest_test

import (
	"testing"

	"github.com/RaduBerinde/tdigest"
)

func TestCentroid_Add(t *testing.T) {
	tests := []struct {
		name    string
		c       tdigest.Centroid
		r       tdigest.Centroid
		want    tdigest.Centroid
		wantErr bool
		errStr  string
	}{
		{
			name: "error when weight is zero",
			r: tdigest.Centroid{
				Weight: -1.0,
			},
			wantErr: true,
			errStr:  "centroid weight cannot be less than zero",
		},
		{
			name: "zero weight",
			c: tdigest.Centroid{
				Weight: 0.0,
				Mean:   1.0,
			},
			r: tdigest.Centroid{
				Weight: 1.0,
				Mean:   2.0,
			},
			want: tdigest.Centroid{
				Weight: 1.0,
				Mean:   2.0,
			},
		},
		{
			name: "weight order of magnitude",
			c: tdigest.Centroid{
				Weight: 1,
				Mean:   1,
			},
			r: tdigest.Centroid{
				Weight: 10,
				Mean:   10,
			},
			want: tdigest.Centroid{
				Weight: 11,
				Mean:   9.181818181818182,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &tt.c
			if err := c.Add(tt.r); (err != nil) != tt.wantErr {
				t.Errorf("Centroid.Add() error = %v, wantErr %v", err, tt.wantErr)
			} else if tt.wantErr && err.Error() != tt.errStr {
				t.Errorf("Centroid.Add() error.Error() = %s, errStr %v", err.Error(), tt.errStr)
			}
			if tt.c != tt.want {
				t.Errorf("unexprected centroid want %f  got %f", tt.want, tt.c)
			}
		})
	}
}
