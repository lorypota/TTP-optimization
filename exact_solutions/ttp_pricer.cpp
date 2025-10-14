#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <map>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <array>
#include <queue>

namespace py = pybind11;

constexpr int MAX_OPPONENTS = 10; 
using Mask = std::array<uint8_t, MAX_OPPONENTS>;
using State = std::tuple<int, int, Mask, int, int>;

// Custom hash and equality for using State in unordered_map
struct StateHash {
    std::size_t operator()(const State& s) const {
        std::size_t h = std::hash<int>()(std::get<0>(s));
        h ^= std::hash<int>()(std::get<1>(s)) << 1;
        h ^= std::hash<int>()(std::get<3>(s)) << 2;
        h ^= std::hash<int>()(std::get<4>(s)) << 3;
        const auto& mask = std::get<2>(s);
        for (size_t i = 0; i < mask.size(); ++i) { h ^= std::hash<uint8_t>()(mask[i]) << (i % 4); }
        return h;
    }
};
struct StateEqual {
    bool operator()(const State& a, const State& b) const {
        return std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b) &&
               std::get<2>(a) == std::get<2>(b) && std::get<3>(a) == std::get<3>(b) &&
               std::get<4>(a) == std::get<4>(b);
    }
};

// Generic solver function, now with a phase parameter
std::optional<std::map<std::string, py::object>>
_solve_pricing_internal(
    int team_idx, const std::map<std::string, py::array_t<double>>& duals,
    const py::array_t<double>& dist_matrix_np, int n, int m_slots,
    const std::map<std::pair<int, int>, int>& pair_to_idx, int phase)
{
    auto dist_buf = dist_matrix_np.request();
    double *dist_ptr = static_cast<double *>(dist_buf.ptr);
    auto dist = [&](int r, int c) { return dist_ptr[r * n + c]; };

    auto duals_team_buf = duals.at("team").request();
    double *duals_team_ptr = static_cast<double *>(duals_team_buf.ptr);
    auto duals_pair_slot_buf = duals.at("pair_slot").request();
    double *duals_pair_slot_ptr = static_cast<double *>(duals_pair_slot_buf.ptr);
    auto duals_home_cons_buf = duals.at("home_cons").request();
    double *duals_home_cons_ptr = static_cast<double *>(duals_home_cons_buf.ptr);
    
    std::vector<int> opponents; std::map<int, int> opp_map;
    int opp_counter = 0;
    for (int i = 0; i < n; ++i) { if (i != team_idx) { opponents.push_back(i); opp_map[i] = opp_counter++; }}
    int num_opp = opponents.size();
    if (num_opp > MAX_OPPONENTS) { throw std::runtime_error("Number of opponents exceeds MAX_OPPONENTS."); }
    
    Mask initial_mask{};
    State source_node = {-1, team_idx, initial_mask, -1, 0};
    using PathInfo = std::tuple<double, State, std::pair<int, bool>>;
    std::unordered_map<State, PathInfo, StateHash, StateEqual> labels;
    using PQElement = std::pair<double, State>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    labels[source_node] = {0.0, source_node, {-1, false}};
    pq.push({0.0, source_node});

    State best_final_node = {-1, -1, {}, -1, -1};
    double min_rc = (phase == 0) ? std::numeric_limits<double>::infinity() : 1e-6; // Phase 0 is initial generation

    while (!pq.empty()) {
        auto [cost, current_node] = pq.top();
        pq.pop();
        if (cost > std::get<0>(labels[current_node])) { continue; }
        auto& [slot, prev_loc, prev_mask, prev_opp, prev_streak] = current_node;
        int s = slot + 1;
        if (s >= m_slots) continue;

        for (int opp_team : opponents) {
            if (opp_team == prev_opp) continue;
            int opp_idx = opp_map.at(opp_team);
            std::vector<std::pair<bool, int>> possible_moves;
            if (prev_mask[opp_idx] == 0) { possible_moves.push_back({true, 1}); possible_moves.push_back({false, 2}); }
            else if (prev_mask[opp_idx] == 1) { possible_moves.push_back({false, 3}); }
            else if (prev_mask[opp_idx] == 2) { possible_moves.push_back({true, 3}); }

            for (const auto& [is_home, new_mask_val] : possible_moves) {
                if ((is_home && prev_streak == 3) || (!is_home && prev_streak == -3)) continue;
                int new_streak = is_home ? (prev_streak > 0 ? prev_streak + 1 : 1) : (prev_streak < 0 ? prev_streak - 1 : -1);
                int venue = is_home ? team_idx : opp_team;
                Mask new_mask = prev_mask; new_mask[opp_idx] = new_mask_val;
                
                // <<< DEFINITIVE FIX: Arc cost depends on the phase >>>
                double arc_cost = (phase == 2) ? dist(prev_loc, venue) : 0.0;

                int i = std::min(team_idx, opp_team), j = std::max(team_idx, opp_team);
                int pair_idx = pair_to_idx.at({i, j});
                int relative_idx = pair_idx * m_slots + s;
                if (s == 0) arc_cost -= duals_team_ptr[team_idx];
                double sign = (team_idx == i) ? 1.0 : -1.0;
                arc_cost -= sign * duals_pair_slot_ptr[relative_idx];
                double hf = is_home ? 1.0 : 0.0;
                if (team_idx == i) arc_cost -= (hf - 1.0) * duals_home_cons_ptr[relative_idx];
                else arc_cost -= hf * duals_home_cons_ptr[relative_idx];
                
                State new_node = {s, venue, new_mask, opp_team, new_streak};
                double new_cost = cost + arc_cost;

                if (labels.find(new_node) == labels.end() || new_cost < std::get<0>(labels[new_node])) {
                    labels[new_node] = {new_cost, current_node, {opp_team, is_home}};
                    pq.push({new_cost, new_node});
                }
            }
        }
    }
    
    Mask final_mask{}; std::fill(final_mask.begin(), final_mask.begin() + num_opp, 3);
    for (const auto& [node, path_info] : labels) {
        const auto& [s_node, loc, mask, _, __] = node;
        if (s_node == m_slots - 1 && mask == final_mask) {
            double final_cost = std::get<0>(path_info);
            if (phase == 2) { final_cost += dist(loc, team_idx); }
            if (final_cost < min_rc) { min_rc = final_cost; best_final_node = node; }
        }
    }
    
    if (std::get<0>(best_final_node) != -1) {
        std::vector<int> opp_seq(m_slots); std::vector<bool> home_flags(m_slots);
        State curr = best_final_node;
        while (std::get<0>(curr) != -1) {
            const auto& [_, __, move] = labels.at(curr);
            int s = std::get<0>(curr);
            opp_seq[s] = move.first; home_flags[s] = move.second;
            curr = std::get<1>(labels.at(curr));
        }
        double true_cost = 0.0; int prev_loc = team_idx;
        for (int s = 0; s < m_slots; ++s) {
            int venue = home_flags[s] ? team_idx : opp_seq[s];
            true_cost += dist(prev_loc, venue); prev_loc = venue;
        }
        true_cost += dist(prev_loc, team_idx);
        py::array_t<int> delta_np({n, m_slots});
        auto delta_buf = delta_np.request();
        int *delta_ptr = static_cast<int *>(delta_buf.ptr);
        std::fill(delta_ptr, delta_ptr + n * m_slots, 0);
        for(int s = 0; s < m_slots; ++s) { delta_ptr[opp_seq[s] * m_slots + s] = 1; }
        std::map<std::string, py::object> tour;
        tour["opp_seq"] = py::cast(opp_seq); tour["home_flags"] = py::cast(home_flags);
        tour["cost"] = py::cast(true_cost); tour["delta"] = delta_np;
        return tour;
    }
    return std::nullopt;
}

PYBIND11_MODULE(ttp_pricer, m) {
    m.doc() = "High-performance C++ pricer for the TTP";
    m.def("solve", &_solve_pricing_internal, "Solves the pricing subproblem",
          py::arg("team_idx"), py::arg("duals"), py::arg("dist_matrix"),
          py::arg("n"), py::arg("m_slots"), py::arg("pair_to_idx"),
          py::arg("phase") = 2);
}

