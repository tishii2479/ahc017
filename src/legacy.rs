pub fn create_initial_state1(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State::new(input.d, vec![INF as usize; input.m], 0);
    for i in 0..input.m {
        let mut day = rnd::gen_range(0, input.d);
        while state.repair_counts[day] >= input.k {
            day = rnd::gen_range(0, input.d);
        }
        state.update_when(i, day);
    }
    state
}

pub fn create_initial_state2(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State::new(input.d, vec![0; input.m], 0);

    for day in 1..input.d {
        let max_count = input.m / input.d;
        let mut count = 0;

        while count < max_count && time::elapsed_seconds() < time_limit {
            let mut s = 0;
            while state.when[s] != 0 {
                s = rnd::gen_range(0, input.m);
            }

            let start_edge = graph.edges[s];
            state.update_when(s, day);
            if !graph.is_connected(&state.when, day) {
                state.update_when(s, 0);
                continue;
            }
            count += 1;

            let mut v = start_edge.v;

            while count < max_count {
                let mut is_added = false;
                for e in &graph.adj[v] {
                    if state.when[e.index] != 0 {
                        continue;
                    }
                    let next_v = e.to;
                    let sim = calc_cosine_similarity(
                        &graph.pos[start_edge.v],
                        &graph.pos[start_edge.u],
                        &graph.pos[next_v],
                        &graph.pos[v],
                    );
                    if sim >= 0.6 {
                        state.update_when(e.index, day);
                        if !graph.is_connected(&state.when, day) {
                            state.update_when(e.index, 0);
                            continue;
                        }
                        v = next_v;
                        count += 1;
                        is_added = true;
                        break;
                    }
                }
                if !is_added {
                    break;
                }
            }
        }
    }
    state
}

pub fn create_initial_state3(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut paths = vec![];
    let mut path_when = vec![];
    for i in 1..input.n {
        for j in 0..i {
            paths.push(graph.get_path(i, j));
            path_when.push(INF as usize);
        }
    }

    let mut ps = vec![];
    for _ in 0..5 {
        ps.push(rnd::gen_range(0, input.n));
    }

    let mut use_path_indices: Vec<usize>;
    let mut state;

    loop {
        // paths.sort_by(|a, b| b.len().partial_cmp(&a.len()).unwrap());
        rnd::shuffle(&mut paths);

        use_path_indices = vec![];

        state = State::new(input.d, vec![INF as usize; input.m], 0);
        for (path_index, path) in paths.iter().enumerate() {
            let mut is_occupied = false;
            for edge_index in path {
                if state.when[*edge_index] != INF as usize {
                    is_occupied = true;
                }
            }
            if is_occupied {
                continue;
            }

            let day = min_index(&state.repair_counts);

            let mut is_encased = false;
            for edge_index in path {
                state.update_when(*edge_index, day);
                if graph.is_encased(&state.when, graph.edges[*edge_index].u)
                    || graph.is_encased(&state.when, graph.edges[*edge_index].v)
                {
                    is_encased = true;
                }
            }
            if is_encased {
                for edge_index in path {
                    state.update_when(*edge_index, INF as usize);
                }
                continue;
            }
            use_path_indices.push(path_index);
            path_when[path_index] = day;
        }

        for (path_index, path) in paths.iter().enumerate() {
            if path.len() != 1 {
                continue;
            }
            let edge_index = path[0];
            if state.when[edge_index] == INF as usize {
                let mut day = rnd::gen_range(0, input.d);
                state.update_when(edge_index, day);
                while graph.is_encased(&state.when, graph.edges[edge_index].u)
                    || graph.is_encased(&state.when, graph.edges[edge_index].v)
                {
                    day = rnd::gen_range(0, input.d);
                    state.update_when(edge_index, day);
                }
                use_path_indices.push(path_index);
                path_when[path_index] = day;
            }
        }

        let mut is_connected = true;
        for day in 0..input.d {
            if !graph.is_connected(&state.when, day) {
                is_connected = false;
            }
        }
        state.score = 0;
        for day in 0..input.d {
            for s in &ps {
                state.score += graph.calc_dist_sum(*s, &state.when, day);
            }
        }
        eprintln!(
            "{}, {}, {}",
            use_path_indices.len(),
            is_connected,
            time::elapsed_seconds()
        );
        if is_connected || time::elapsed_seconds() >= time_limit {
            break;
        }
    }

    state.score = 0;
    for day in 0..input.d {
        for s in &ps {
            state.score += graph.calc_dist_sum(*s, &state.when, day);
        }
    }

    state
}
