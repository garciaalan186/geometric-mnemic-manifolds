#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>

// Simple struct for Top-K results
struct Result {
    int id;
    float score;

    bool operator>(const Result& other) const {
        return score > other.score; // Min-heap for keeping largest elements
    }
};

// Compute dot product of two vectors
// Clang will auto-vectorize this loop heavily with -O3 -march=native
inline float dot_product(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Fast Linear Scan
// data: flattened array of shape (N, D)
// query: array of shape (D)
// indices: array of shape (N) - corresponding IDs
// k: number of top results to return
std::vector<std::pair<float, int>> fast_scan(
    const float* data, 
    int n, 
    int d, 
    const int* indices, 
    const float* query, 
    int k
) {
    // Min-priority queue to maintain top-k largest scores
    // We store pair<score, id>
    // Priority Key is score. std::priority_queue is Max-Heap by default.
    // To implement a "Top-K" we usually use a Min-Heap of size K. 
    // If new_score > min_heap.top(), we pop min and push new.
    
    std::vector<std::pair<float, int>> top_k;
    top_k.reserve(k + 1);
    
    // We will use standard vector sort/heap operations for cache efficiency over std::priority_queue
    // Actually, for moderate K, maintaining a heap is fine.
    
    using Entry = std::pair<float, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> min_heap;

    for (int i = 0; i < n; ++i) {
        // Calculate similarity
        const float* vec = data + (i * d);
        float score = dot_product(vec, query, d);
        
        if (min_heap.size() < k) {
            min_heap.push({score, indices[i]});
        } else if (score > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({score, indices[i]});
        }
    }

    // Convert heap to sorted vector (descending score)
    int result_size = min_heap.size();
    std::vector<Entry> results(result_size);
    for (int i = result_size - 1; i >= 0; --i) {
        results[i] = min_heap.top();
        min_heap.pop();
    }
    
    return results;
}
