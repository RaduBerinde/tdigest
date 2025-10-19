package tdigest

import (
	"math"
	"math/bits"
)

// TDigest is a data structure for accurate on-line accumulation of
// rank-based statistics such as quantiles and trimmed means.
type TDigest struct {
	Compression float64

	maxProcessed      int
	maxUnprocessed    int
	processed         CentroidList
	unprocessed       CentroidList
	processedWeight   float64
	unprocessedWeight float64
	min               float64
	max               float64
}

// New initializes a new distribution with a default compression.
func New() *TDigest {
	return NewWithCompression(1000)
}

// NewWithCompression initializes a new distribution with custom compression.
//
// Delta is the compression factor, which determines the size of the digest: it
// contains at most 2*delta centroids (but closer to 1.3*delta in practice).
func NewWithCompression(delta int) *TDigest {
	t := &TDigest{
		Compression:    float64(delta),
		maxProcessed:   2 * delta,
		maxUnprocessed: unprocessedFactor * 2 * delta,
	}
	t.processed = make(CentroidList, 0, t.maxProcessed)
	t.unprocessed = make(CentroidList, 0, t.maxUnprocessed+t.maxProcessed+1)
	t.Reset()
	return t
}

// unprocessedFactor is the relative size of accumulated samples before a merge
// and recompress step. The goal of this value is to balance the cost of the
// sort against the cost of evaluating the scale function.
const unprocessedFactor = 2

// Reset resets the distribution to its initial state.
func (t *TDigest) Reset() {
	t.processed = t.processed[:0]
	t.unprocessed = t.unprocessed[:0]
	t.processedWeight = 0
	t.unprocessedWeight = 0
	t.min = math.MaxFloat64
	t.max = -math.MaxFloat64
}

// Add adds a value x with a weight w to the distribution.
func (t *TDigest) Add(x, w float64) {
	t.AddCentroid(Centroid{Mean: x, Weight: w})
}

// AddCentroidList can quickly add multiple centroids.
func (t *TDigest) AddCentroidList(c CentroidList) {
	// It's possible to optimize this by bulk-copying the slice, but this
	// yields just a 1-2% speedup (most time is in process()), so not worth
	// the complexity.
	for i := range c {
		t.AddCentroid(c[i])
	}
}

// AddCentroid adds a single centroid.
// Weights which are not a number or are <= 0 are ignored, as are NaN means.
func (t *TDigest) AddCentroid(c Centroid) {
	if math.IsNaN(c.Mean) || c.Weight <= 0 || math.IsNaN(c.Weight) || math.IsInf(c.Weight, 1) {
		return
	}

	t.unprocessed = append(t.unprocessed, c)
	t.unprocessedWeight += c.Weight

	if t.processed.Len() > t.maxProcessed ||
		t.unprocessed.Len() > t.maxUnprocessed {
		t.process()
	}
}

// Merge the supplied digest into this digest. Functionally equivalent to
// calling t.AddCentroidList(t2.Centroids(nil)), but avoids making an extra copy
// of the CentroidList.
func (t *TDigest) Merge(t2 *TDigest) {
	t2.process()
	t.AddCentroidList(t2.processed)
}

func (t *TDigest) process() {
	if t.unprocessed.Len() > 0 ||
		t.processed.Len() > t.maxProcessed {

		// Append all processed centroids to the unprocessed list and sort
		n := len(t.unprocessed)
		t.unprocessed = append(t.unprocessed, t.processed...)
		//pdqsort(t.unprocessed, 0, len(t.unprocessed), bits.Len(uint(len(t.unprocessed))))
		pdqsort(t.unprocessed, 0, n, bits.Len(uint(n)))

		// Reset processed list with first centroid
		t.processed.Clear()

		m := merger{a: t.unprocessed[n:], b: t.unprocessed[:n]}
		centroid, _ := m.Next()
		t.processed = append(t.processed, centroid)

		t.processedWeight += t.unprocessedWeight
		t.unprocessedWeight = 0
		soFar := centroid.Weight
		limit := t.processedWeight * t.integratedQ(1.0)
		for {
			centroid, ok := m.Next()
			if !ok {
				break
			}

			projected := soFar + centroid.Weight
			if projected <= limit {
				soFar = projected
				_ = t.processed[t.processed.Len()-1].Add(centroid)
			} else {
				k1 := t.integratedLocation(soFar / t.processedWeight)
				limit = t.processedWeight * t.integratedQ(k1+1.0)
				soFar += centroid.Weight
				t.processed = append(t.processed, centroid)
			}
		}
		t.min = math.Min(t.min, t.processed[0].Mean)
		t.max = math.Max(t.max, t.processed[t.processed.Len()-1].Mean)
		t.unprocessed.Clear()
	}
}

type merger struct {
	a, b CentroidList
}

func (m *merger) Next() (Centroid, bool) {
	if len(m.a) > 0 {
		if len(m.b) == 0 || m.a[0].Mean <= m.b[0].Mean {
			c := m.a[0]
			m.a = m.a[1:]
			return c, true
		}
	} else if len(m.b) == 0 {
		return Centroid{}, false
	}
	c := m.b[0]
	m.b = m.b[1:]
	return c, true
}

// Centroids returns a copy of processed centroids.
// Useful when aggregating multiple t-digests.
//
// Centroids are appended to the passed CentroidList; if you're re-using a
// buffer, be sure to pass cl[:0].
func (t *TDigest) Centroids(cl CentroidList) CentroidList {
	t.process()
	return append(cl, t.processed...)
}

func (t *TDigest) Count() float64 {
	t.process()

	// t.process always updates t.processedWeight to the total count of all
	// centroids, so we don't need to re-count here.
	return t.processedWeight
}

// Quantile returns the (approximate) quantile of
// the distribution. Accepted values for q are between 0.0 and 1.0.
// Returns NaN if Count is zero or bad inputs.
func (t *TDigest) Quantile(q float64) float64 {
	t.process()
	if q < 0 || q > 1 || t.processed.Len() == 0 {
		return math.NaN()
	}
	if t.processed.Len() == 1 {
		return t.processed[0].Mean
	}
	index := q * t.processedWeight
	if index <= t.processed[0].Weight/2.0 {
		return t.min + 2.0*index/t.processed[0].Weight*(t.processed[0].Mean-t.min)
	}

	var sum, cumulative, prevCumulative float64
	for i := range t.processed {
		cur := t.processed[i].Weight
		prevCumulative = cumulative
		// cumulative is the sum of all previous weights plus half of this
		// centroid's weight.
		cumulative = sum + cur/2.0
		sum += cur
		if cumulative >= index {
			z1 := index - prevCumulative
			z2 := cumulative - index
			return weightedAverage(t.processed[i-1].Mean, z2, t.processed[i].Mean, z1)
		}
	}
	z1 := index - t.processedWeight - t.processed[len(t.processed)-1].Weight/2.0
	z2 := (t.processed[len(t.processed)-1].Weight / 2.0) - z1
	return weightedAverage(t.processed[len(t.processed)-1].Mean, z1, t.max, z2)
}

// CDF returns the cumulative distribution function for a given value x.
func (t *TDigest) CDF(x float64) float64 {
	t.process()
	switch t.processed.Len() {
	case 0:
		return 0.0
	case 1:
		width := t.max - t.min
		if x <= t.min {
			return 0.0
		}
		if x >= t.max {
			return 1.0
		}
		if (x - t.min) <= width {
			// min and max are too close together to do any viable interpolation
			return 0.5
		}
		return (x - t.min) / width
	}

	if x <= t.min {
		return 0.0
	}
	if x >= t.max {
		return 1.0
	}
	m0 := t.processed[0].Mean
	// Left Tail
	if x <= m0 {
		if m0-t.min > 0 {
			return (x - t.min) / (m0 - t.min) * t.processed[0].Weight / t.processedWeight / 2.0
		}
		return 0.0
	}

	var sum, cumulative, prevCumulative float64
	for i := range t.processed {
		cur := t.processed[i].Weight
		prevCumulative = cumulative
		// cumulative is the sum of all previous weights plus half of this
		// centroid's weight.
		cumulative = sum + cur/2.0
		sum += cur
		if t.processed[i].Mean > x {
			z1 := x - t.processed[i-1].Mean
			z2 := t.processed[i].Mean - x
			return weightedAverage(prevCumulative, z2, cumulative, z1) / t.processedWeight
		}
	}

	// Right Tail
	mn := t.processed[t.processed.Len()-1].Mean
	if t.max-mn > 0.0 {
		return 1.0 - (t.max-x)/(t.max-mn)*t.processed[t.processed.Len()-1].Weight/t.processedWeight/2.0
	}
	return 1.0
}

func (t *TDigest) integratedQ(k float64) float64 {
	return (math.Sin(math.Min(k, t.Compression)*math.Pi/t.Compression-math.Pi/2.0) + 1.0) / 2.0
}

func (t *TDigest) integratedLocation(q float64) float64 {
	return t.Compression * (math.Asin(2.0*q-1.0) + math.Pi/2.0) / math.Pi
}

func weightedAverage(x1, w1, x2, w2 float64) float64 {
	if x1 <= x2 {
		return weightedAverageSorted(x1, w1, x2, w2)
	}
	return weightedAverageSorted(x2, w2, x1, w1)
}

func weightedAverageSorted(x1, w1, x2, w2 float64) float64 {
	x := (x1*w1 + x2*w2) / (w1 + w2)
	return math.Max(x1, math.Min(x, x2))
}
