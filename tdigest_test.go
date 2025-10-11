package tdigest_test

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"sort"
	"testing"

	"github.com/RaduBerinde/tdigest"
)

const (
	N     = 1e6
	Mu    = 10
	Sigma = 3

	seed = 42
)

// NormalData is a slice of N random values that are normaly distributed with mean Mu and standard deviation Sigma.
var NormalData []float64
var UniformData []float64

var NormalDigest *tdigest.TDigest
var UniformDigest *tdigest.TDigest

func init() {
	UniformData = make([]float64, N)
	UniformDigest = tdigest.NewWithCompression(1000)
	rng := rand.New(rand.NewPCG(seed, seed))
	for i := range UniformData {
		UniformData[i] = rng.Float64() * 100
		UniformDigest.Add(UniformData[i], 1)
	}

	NormalData = make([]float64, N)
	NormalDigest = tdigest.NewWithCompression(1000)
	rng = rand.New(rand.NewPCG(seed, seed))
	for i := range NormalData {
		NormalData[i] = rng.NormFloat64()*Sigma + Mu
		NormalDigest.Add(NormalData[i], 1)
	}
}

// Compares the quantile results of two digests, and fails if the
// fractional err exceeds maxErr.
// Always fails if the total count differs.
func compareQuantiles(td1, td2 *tdigest.TDigest, maxErr float64) error {
	if td1.Count() != td2.Count() {
		return fmt.Errorf("counts are not equal, %d vs %d", int64(td1.Count()), int64(td2.Count()))
	}
	for q := 0.05; q < 1; q += 0.05 {
		if math.Abs(td1.Quantile(q)-td2.Quantile(q))/td1.Quantile(q) > maxErr {
			return fmt.Errorf("quantile %g differs, %g vs %g", q, td1.Quantile(q), td2.Quantile(q))
		}
	}
	return nil
}

// All Add methods should yield equivalent results.
func TestTdigest_AddFuncs(t *testing.T) {
	centroids := NormalDigest.Centroids(nil)

	addDigest := tdigest.NewWithCompression(100)
	addCentroidDigest := tdigest.NewWithCompression(100)
	addCentroidListDigest := tdigest.NewWithCompression(100)

	for _, c := range centroids {
		addDigest.Add(c.Mean, c.Weight)
		addCentroidDigest.AddCentroid(c)
	}
	addCentroidListDigest.AddCentroidList(centroids)

	if err := compareQuantiles(addDigest, addCentroidDigest, 0.01); err != nil {
		t.Errorf("AddCentroid() differs from from Add(): %s", err.Error())
	}
	if err := compareQuantiles(addDigest, addCentroidListDigest, 0.01); err != nil {
		t.Errorf("AddCentroidList() differs from from Add(): %s", err.Error())
	}
}

func TestTdigest_Count(t *testing.T) {
	tests := []struct {
		name   string
		data   []float64
		digest *tdigest.TDigest
		want   float64
	}{
		{
			name: "empty",
			data: []float64{},
			want: 0,
		},
		{
			name: "not empty",
			data: []float64{5, 4},
			want: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			td := tt.digest
			if td == nil {
				td = tdigest.NewWithCompression(1000)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
			}
			got := td.Count()
			if got != tt.want {
				t.Errorf("unexpected count, got %g want %g", got, tt.want)
			}
		})
	}

	got := NormalDigest.Count()
	want := float64(len(NormalData))
	if got != want {
		t.Errorf("unexpected count for NormalDigest, got %g want %g", got, want)
	}

	got = UniformDigest.Count()
	want = float64(len(UniformData))
	if got != want {
		t.Errorf("unexpected count for UniformDigest, got %g want %g", got, want)
	}
}

func TestTdigest_Quantile(t *testing.T) {
	const eps = 1e-3
	tests := []struct {
		name     string
		data     []float64
		digest   *tdigest.TDigest
		quantile float64
		want     float64
	}{
		{
			name:     "increasing",
			quantile: 0.5,
			data:     []float64{1, 2, 3, 4, 5},
			want:     3,
		},
		{
			name:     "data in decreasing order",
			quantile: 0.25,
			data:     []float64{555.349107, 432.842597},
			want:     432.842597,
		},
		{
			name:     "small",
			quantile: 0.5,
			data:     []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
			want:     3,
		},
		{
			name:     "small 99 (max)",
			quantile: 0.99,
			data:     []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
			want:     5,
		},
		{
			name:     "normal 50",
			quantile: 0.5,
			digest:   NormalDigest,
			want:     Mu,
		},
		{
			name:     "normal 90",
			quantile: 0.9,
			digest:   NormalDigest,
			want:     Mu + 1.281551565*Sigma,
		},
		{
			name:     "uniform 50",
			quantile: 0.5,
			digest:   UniformDigest,
			want:     50,
		},
		{
			name:     "uniform 90",
			quantile: 0.9,
			digest:   UniformDigest,
			want:     90,
		},
		{
			name:     "uniform 99",
			quantile: 0.99,
			digest:   UniformDigest,
			want:     99,
		},
		{
			name:     "uniform 99.9",
			quantile: 0.999,
			digest:   UniformDigest,
			want:     99.9,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			td := tt.digest
			if td == nil {
				td = tdigest.NewWithCompression(1000)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
			}
			got := td.Quantile(tt.quantile)
			if math.Abs(tt.want-got) > math.Max(eps, eps*tt.want) {
				t.Errorf("unexpected quantile %f, got %g want %g", tt.quantile, got, tt.want)
			}
		})
	}
}

func TestTdigest_CDFs(t *testing.T) {
	const eps = 1e-3
	tests := []struct {
		name   string
		data   []float64
		digest *tdigest.TDigest
		cdf    float64
		want   float64
	}{
		{
			name: "increasing",
			cdf:  3,
			data: []float64{1, 2, 3, 4, 5},
			want: 0.5,
		},
		{
			name: "small",
			cdf:  4,
			data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
			want: 0.75,
		},
		{
			name: "small max",
			cdf:  5,
			data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
			want: 1,
		},
		{
			name: "normal mean",
			cdf:  10,
			data: NormalData,
			want: 0.5,
		},
		{
			name: "normal high",
			cdf:  -100,
			data: NormalData,
			want: 0,
		},
		{
			name: "normal low",
			cdf:  110,
			data: NormalData,
			want: 1,
		},
		{
			name: "uniform 50",
			cdf:  50,
			data: UniformData,
			want: 0.5,
		},
		{
			name: "uniform min",
			cdf:  0,
			data: UniformData,
			want: 0,
		},
		{
			name: "uniform max",
			cdf:  100,
			data: UniformData,
			want: 1,
		},
		{
			name: "uniform 10",
			cdf:  10,
			data: UniformData,
			want: 0.1,
		},
		{
			name: "uniform 90",
			cdf:  90,
			data: UniformData,
			want: 0.9,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			td := tt.digest
			if td == nil {
				td = tdigest.NewWithCompression(1000)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
			}
			got := td.CDF(tt.cdf)
			if math.Abs(tt.want-got) > eps {
				t.Errorf("unexpected CDF %f, got %g want %g", tt.cdf, got, tt.want)
			}
		})
	}
}

func TestTdigest_Reset(t *testing.T) {
	td := tdigest.New()
	for _, x := range NormalData {
		td.Add(x, 1)
	}
	q1 := td.Quantile(0.9)

	td.Reset()
	for _, x := range NormalData {
		td.Add(x, 1)
	}
	if q2 := td.Quantile(0.9); q2 != q1 {
		t.Errorf("unexpected quantile, got %g want %g", q2, q1)
	}
}

func TestTdigest_OddInputs(t *testing.T) {
	td := tdigest.New()
	td.Add(math.NaN(), 1)
	td.Add(1, math.NaN())
	td.Add(1, 0)
	td.Add(1, -1000)
	if td.Count() != 0 {
		t.Error("invalid value was alloed to be added")
	}

	// Infinite values are allowed.
	td.Add(1, 1)
	td.Add(2, 1)
	td.Add(math.Inf(1), 1)
	if q := td.Quantile(0.5); q != 2 {
		t.Errorf("expected median value 2, got %f", q)
	}
	if q := td.Quantile(0.9); !math.IsInf(q, 1) {
		t.Errorf("expected median value 2, got %f", q)
	}
}

func TestTdigest_Merge(t *testing.T) {
	// Repeat merges enough times to ensure we call compress()
	numRepeats := 20
	addDigest := tdigest.New()
	for i := 0; i < numRepeats; i++ {
		for _, c := range NormalDigest.Centroids(nil) {
			addDigest.AddCentroid(c)
		}
		for _, c := range UniformDigest.Centroids(nil) {
			addDigest.AddCentroid(c)
		}
	}

	mergeDigest := tdigest.New()
	for i := 0; i < numRepeats; i++ {
		mergeDigest.Merge(NormalDigest)
		mergeDigest.Merge(UniformDigest)
	}

	if err := compareQuantiles(addDigest, mergeDigest, 0.001); err != nil {
		t.Errorf("AddCentroid() differs from from Merge(): %s", err.Error())
	}

	// Empty merge does nothing and has no effect on underlying centroids.
	c1 := addDigest.Centroids(nil)
	addDigest.Merge(tdigest.New())
	c2 := addDigest.Centroids(nil)
	if !reflect.DeepEqual(c1, c2) {
		t.Error("Merging an empty digest altered data")
	}
}

var quantiles = []float64{0.1, 0.5, 0.9, 0.99, 0.999}

func BenchmarkTDigest_Add(b *testing.B) {
	for n := 0; n < b.N; n++ {
		td := tdigest.NewWithCompression(1000)
		for _, x := range NormalData {
			td.Add(x, 1)
		}
	}
}

func BenchmarkTDigest_AddCentroid(b *testing.B) {
	centroids := make(tdigest.CentroidList, len(NormalData))
	for i := range centroids {
		centroids[i].Mean = NormalData[i]
		centroids[i].Weight = 1
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		td := tdigest.NewWithCompression(1000)
		for i := range centroids {
			td.AddCentroid(centroids[i])
		}
	}
}

func BenchmarkTDigest_AddCentroidList(b *testing.B) {
	centroids := make(tdigest.CentroidList, len(NormalData))
	for i := range centroids {
		centroids[i].Mean = NormalData[i]
		centroids[i].Weight = 1
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		td := tdigest.NewWithCompression(1000)
		td.AddCentroidList(centroids)
	}
}

func BenchmarkTDigest_Merge(b *testing.B) {
	b.Run("AddCentroid", func(b *testing.B) {
		var cl tdigest.CentroidList
		td := tdigest.New()
		for n := 0; n < b.N; n++ {
			cl = NormalDigest.Centroids(cl[:0])
			for i := range cl {
				td.AddCentroid(cl[i])
			}
		}
	})
	b.Run("Merge", func(b *testing.B) {
		td := tdigest.New()
		for n := 0; n < b.N; n++ {
			td.Merge(NormalDigest)
		}
	})
}

func BenchmarkTDigest_Quantile(b *testing.B) {
	td := tdigest.NewWithCompression(1000)
	for _, x := range NormalData {
		td.Add(x, 1)
	}
	b.ResetTimer()
	var x float64
	for n := 0; n < b.N; n++ {
		for _, q := range quantiles {
			x += td.Quantile(q)
		}
	}
}

func TestTdigest_Centroids(t *testing.T) {
	tests := []struct {
		name   string
		data   []float64
		digest *tdigest.TDigest
		want   tdigest.CentroidList
	}{
		{
			name: "increasing",
			data: []float64{1, 2, 3, 4, 5},
			want: tdigest.CentroidList{
				tdigest.Centroid{
					Mean:   1.0,
					Weight: 1.0,
				},

				tdigest.Centroid{
					Mean:   2.5,
					Weight: 2.0,
				},

				tdigest.Centroid{
					Mean:   4.0,
					Weight: 1.0,
				},

				tdigest.Centroid{
					Mean:   5.0,
					Weight: 1.0,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got tdigest.CentroidList
			td := tt.digest
			if td == nil {
				td = tdigest.NewWithCompression(3)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
			}
			got = td.Centroids(got[:0])
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("unexpected list got %g want %g", got, tt.want)
			}
		})
	}
}

type sample struct {
	Value  float64
	Weight float64
}

func makeSamples(values, weights []float64) []sample {
	s := make([]sample, len(values))
	for i := range s {
		s[i].Value = values[i]
		if weights[i] <= 0 {
			panic("weights must be positive")
		}
		s[i].Weight = weights[i]
	}
	return s
}

// expectedFns returns functions to compute the exact quantile and CDF for the
// given samples.
func expectedFns(
	values []float64, weights []float64,
) (quantileFn func(quantile float64) float64, cdfFn func(value float64) float64) {
	samples := makeSamples(values, weights)
	samples = slices.Clone(samples)
	slices.SortFunc(samples, func(a, b sample) int {
		return cmp.Compare(a.Value, b.Value)
	})
	cumulativeWeight := make([]float64, len(samples))
	totalWeight := 0.0
	for i := range samples {
		totalWeight += samples[i].Weight
		cumulativeWeight[i] = totalWeight
	}
	quantileFn = func(quantile float64) float64 {
		w := totalWeight * quantile
		idx := sort.SearchFloat64s(cumulativeWeight[:len(samples)-1], w)
		// Invariant: cumulativeWeight[idx-1] < w <= cumulativeWeight[idx]. Choose
		// the closer one.
		if idx > 0 && w < (cumulativeWeight[idx-1]+cumulativeWeight[idx])*0.5 {
			idx--
		}
		return samples[idx].Value
	}
	cdfFn = func(value float64) float64 {
		idx := sort.Search(len(samples), func(i int) bool {
			return samples[i].Value > value
		})
		if idx == len(samples) {
			return 1.0
		}
		return cumulativeWeight[idx] / totalWeight
	}
	return quantileFn, cdfFn
}

var testDistributions = map[string]func(rng *rand.Rand) float64{
	"constant": func(rng *rand.Rand) float64 {
		return 42
	},
	"normal": func(rng *rand.Rand) float64 {
		return rng.NormFloat64() + 10
	},
	"uniform": func(rng *rand.Rand) float64 {
		return 100 * rng.Float64()
	},
	"exponential": func(rng *rand.Rand) float64 {
		return rng.ExpFloat64() * 10
	},
	"bimodal": func(rng *rand.Rand) float64 {
		if rng.IntN(2) == 0 {
			return rng.NormFloat64() + 10
		} else {
			return rng.NormFloat64()*2 + 20
		}
	},
}

var testDistributionModifiers = map[string]func(rng *rand.Rand, values []float64){
	"": func(rng *rand.Rand, values []float64) {},
	"-mostly-sorted": func(rng *rand.Rand, values []float64) {
		slices.Sort(values)
		for range len(values) / 5 {
			i := rng.IntN(len(values))
			j := rng.IntN(len(values))
			values[i], values[j] = values[j], values[i]
		}
	},
	"-mostly-revsorted": func(rng *rand.Rand, values []float64) {
		slices.Sort(values)
		slices.Reverse(values)
		for range len(values) / 5 {
			i := rng.IntN(len(values))
			j := rng.IntN(len(values))
			values[i], values[j] = values[j], values[i]
		}
	},
}

func TestAccuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	for _, delta := range []int{100, 1000} {
		t.Run(fmt.Sprintf("delta=%v", delta), func(t *testing.T) {
			for valDistName, valDistFn := range testDistributions {
				for valModName, valModFn := range testDistributionModifiers {
					if valDistName == "constant" && valModName != "" {
						continue
					}
					t.Run(fmt.Sprintf("%s%s", valDistName, valModName), func(t *testing.T) {
						for weightDistName, weightDistFn := range testDistributions {
							for weightModName, weightModFn := range testDistributionModifiers {
								if weightDistName == "constant" && weightModName != "" {
									continue
								}
								t.Run(fmt.Sprintf("%s%s", weightDistName, weightModName), func(t *testing.T) {
									for _, n := range []int{10_000, 1_000_000} {
										t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
											seed := rand.Uint64()
											t.Logf("seed: %d", seed)
											rng := rand.New(rand.NewPCG(seed, seed))
											values := make([]float64, n)
											weights := make([]float64, n)
											for i := range values {
												values[i] = valDistFn(rng)
												weights[i] = max(1e-5, weightDistFn(rng))
											}
											valModFn(rng, values)
											weightModFn(rng, weights)
											trueQuantile, trueCDF := expectedFns(values, weights)

											s := tdigest.NewWithCompression(delta)
											for i := range values {
												s.Add(values[i], weights[i])
											}

											var maxRankErr, maxRankErrTail float64
											for p := range 101 {
												q := float64(p) / 100.0

												// Test Quantile accuracy.
												x := s.Quantile(q)
												if valDistName == "constant" {
													if x != values[0] {
														t.Fatalf("constant value disitibution, expected %v, got %v", values[0], x)
													}
													continue
												}

												xQuantile := trueCDF(x)
												err := math.Abs(q - xQuantile)
												maxRankErr = max(maxRankErr, err)
												if p <= 10 || p >= 90 {
													maxRankErrTail = max(maxRankErrTail, err)
												}
												const debug = false
												if debug {
													t.Logf("%3d%%: x=%.3f  xQuantile=%.2f%%   err=%.2f", p, x, xQuantile*100, err*100)
												}

												// Test CDF accuracy.
												y := trueQuantile(q)
												yQuantile := s.CDF(y)
												err = math.Abs(q - yQuantile)
												maxRankErr = max(maxRankErr, err)
												if p <= 10 || p >= 90 {
													maxRankErrTail = max(maxRankErrTail, err)
												}
												if debug {
													t.Logf("%3d%%: y=%.3f  yQuantile=%.2f%%   err=%.2f", p, y, yQuantile*100, err*100)
												}
											}
											t.Logf("max rank error: %.2f%%", maxRankErr*100)
											t.Logf("max rank error for tail: %.2f%%", maxRankErrTail*100)
											// Reasonable cutoffs for max rank error are 1% for n=100
											// and 0.3% for n=1000. Note that in practice we usually
											// see less than 0.1% for n=1000 but we sometimes seee
											// errors around 0.2%.
											cutoff := 0.01
											if n == 1000 {
												cutoff = 0.003
											}
											if maxRankErr > cutoff {
												t.Fatalf("rank error %.2f%% too high", maxRankErr*100)
											}
										})
									}
								})
							}
						}
					})
				}
			}
		})
	}
}
