// Exact ordered triangle counter for validating DOVE orders at scale.
// Input graph: whitespace-separated integer edge list. Input order: one
// original vertex id per line, from earliest to latest.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: exact_triangle_lister GRAPH.edgelist ORDER.txt [REPETITIONS]\n";
    return 2;
  }
  const int repetitions = argc == 4 ? std::stoi(argv[3]) : 1;
  if (repetitions < 1) throw std::runtime_error("repetitions must be positive");
  std::ifstream input(argv[1]);
  if (!input) throw std::runtime_error("cannot open graph");
  std::vector<std::pair<long long, long long>> raw_edges;
  std::vector<long long> ids;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '%') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream stream(line);
    long long u, v;
    if (!(stream >> u >> v) || u == v) continue;
    if (u > v) std::swap(u, v);
    raw_edges.emplace_back(u, v);
    ids.push_back(u);
    ids.push_back(v);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  std::unordered_map<long long, int> dense;
  dense.reserve(ids.size() * 2);
  for (int i = 0; i < static_cast<int>(ids.size()); ++i) dense[ids[i]] = i;
  std::vector<std::pair<int, int>> edges;
  edges.reserve(raw_edges.size());
  for (auto [u, v] : raw_edges) edges.emplace_back(dense.at(u), dense.at(v));
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::ifstream order_input(argv[2]);
  if (!order_input) throw std::runtime_error("cannot open order");
  std::vector<int> rank(ids.size(), -1);
  long long original;
  int position = 0;
  while (order_input >> original) {
    auto found = dense.find(original);
    if (found == dense.end() || rank[found->second] != -1)
      throw std::runtime_error("order contains an unknown or duplicate id");
    rank[found->second] = position++;
  }
  if (position != static_cast<int>(ids.size()))
    throw std::runtime_error("order does not contain every graph vertex");

  std::vector<std::vector<int>> forward(ids.size()), incoming(ids.size());
  std::vector<int> degree(ids.size(), 0);
  for (auto [u, v] : edges) {
    ++degree[u];
    ++degree[v];
    if (rank[u] < rank[v]) {
      forward[u].push_back(v);
      incoming[v].push_back(u);
    } else {
      forward[v].push_back(u);
      incoming[u].push_back(v);
    }
  }
  // AOT's degree-descending local order improves bitmap locality without
  // changing either output or the candidate-probe count.
  const auto local_order = [&degree](int u, int v) {
    return degree[u] > degree[v] || (degree[u] == degree[v] && u < v);
  };
  for (std::size_t u = 0; u < forward.size(); ++u) {
    std::sort(forward[u].begin(), forward[u].end(), local_order);
    std::sort(incoming[u].begin(), incoming[u].end(), local_order);
  }
  std::uint64_t expected_probes = 0, expected_triangles = 0;
  std::vector<double> timings;
  timings.reserve(repetitions);
  std::vector<int> mark(ids.size(), -1);
  const auto has_smaller_forward_degree = [&forward](int u, int v) {
    return forward[u].size() < forward[v].size()
        || (forward[u].size() == forward[v].size() && u < v);
  };
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    std::uint64_t probes = 0, triangles = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int u = 0; u < static_cast<int>(forward.size()); ++u) {
      for (int w : forward[u]) mark[w] = u;
      for (int v : forward[u]) {
        if (!has_smaller_forward_degree(v, u)) continue;
        probes += forward[v].size();
        for (int w : forward[v]) triangles += mark[w] == u;
      }
      for (int x : incoming[u]) {
        if (!has_smaller_forward_degree(x, u)) continue;
        probes += forward[x].size();
        for (int y : forward[x]) triangles += mark[y] == u;
      }
    }
    const auto stopped = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double>(stopped - started).count());
    if (repetition == 0) {
      expected_probes = probes;
      expected_triangles = triangles;
    } else if (probes != expected_probes || triangles != expected_triangles) {
      throw std::runtime_error("non-deterministic repeated result");
    }
  }
  const double mean = std::accumulate(timings.begin(), timings.end(), 0.0)
      / static_cast<double>(timings.size());
  std::vector<double> sorted = timings;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted.size() % 2
      ? sorted[sorted.size() / 2]
      : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
  std::cout << "{\"vertices\":" << ids.size() << ",\"edges\":" << edges.size()
            << ",\"candidate_probes\":" << expected_probes << ",\"triangles\":"
            << expected_triangles << ",\"repetitions\":" << repetitions
            << ",\"listing_seconds_median\":" << median
            << ",\"listing_seconds_mean\":" << mean
            << ",\"listing_seconds_min\":" << sorted.front() << "}\n";
}
