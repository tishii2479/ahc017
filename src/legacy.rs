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

pub fn optimize_state_slow(
    state: &mut State,
    input: &Input,
    graph: &Graph,
    time_limit: f64,
    debug: bool,
) {
    eprintln!("before: {}", calc_actual_score_slow(&input, &graph, &state));
    let ps = vec![
        graph.find_closest_point(&Pos { x: 250, y: 250 }),
        graph.find_closest_point(&Pos { x: 250, y: 750 }),
        // graph.find_closest_point(&Pos { x: 500, y: 500 }),
        graph.find_closest_point(&Pos { x: 750, y: 250 }),
        graph.find_closest_point(&Pos { x: 750, y: 750 }),
    ];
    // eprintln!("{:?}", ps);

    state.score = 0.;
    for day in 0..input.d {
        for s in &ps {
            state.score += graph.calc_dist_sum(*s, &state.when, day) as f64;
        }
    }

    let mut score_progress_file = File::create("out/optimize_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let edge_index = rnd::gen_range(0, input.m);
        // TODO: 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let prev = state.when[edge_index];
        let next = rnd::gen_range(0, input.d);

        let mut new_score = state.score;

        // TODO: キャッシュする
        for s in &ps {
            new_score -= graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score -= graph.calc_dist_sum(*s, &state.when, next) as f64;
        }
        state.update_when(edge_index, next);
        for s in &ps {
            new_score += graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score += graph.calc_dist_sum(*s, &state.when, next) as f64;
        }

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = new_score < state.score && is_valid;
        if adopt {
            state.score = new_score;
        } else {
            state.update_when(edge_index, prev);
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            if debug {
                writeln!(
                    score_progress_file,
                    "{},{:.2},{}",
                    state.score,
                    time::elapsed_seconds(),
                    calc_actual_score_slow(&input, &graph, &state),
                )
                .unwrap();
            }
            // eprintln!("{} {:.2}", state.score, time::elapsed_seconds());
        }
    }

    eprintln!("[optimize_state] iter_count: {}", iter_count);
}

pub fn create_initial_state(input: &Input, graph: &Graph, time_limit: f64, debug: bool) -> State {
    fn calc_vertex_score(v: usize, graph: &Graph, state: &State) -> f64 {
        let mut score = 0.;
        for e1 in &graph.adj[v] {
            for e2 in &graph.adj[v] {
                if e1.index == e2.index {
                    continue;
                }
                if state.when[e1.index] != state.when[e2.index] {
                    continue;
                }
                let sim = calc_cosine_similarity(
                    &graph.pos[e1.to],
                    &graph.pos[v],
                    &graph.pos[e2.to],
                    &graph.pos[v],
                );
                // let e_score = sim;
                let mut e_score = sim + 0.7;
                if e_score < 0. {
                    e_score *= 3.;
                }
                score += e_score;
            }
        }
        score
    }

    let paths = {
        let mut paths = vec![];
        for i in 1..input.n {
            for j in 0..i {
                let p = graph.get_path(i, j);
                if p.0.len() <= 3 {
                    paths.push(p);
                }
            }
        }
        paths
    };

    let mut state = State::new(input.d, vec![NA; input.m], 0.);
    for i in 0..input.m {
        let mut day = rnd::gen_range(0, input.d);
        while state.repair_counts[day] >= input.k {
            day = rnd::gen_range(0, input.d);
        }
        state.update_when(i, day);
    }

    for v in 0..input.n {
        state.score += calc_vertex_score(v, &graph, &state);
    }

    let mut score_progress_file =
        File::create("out/create_initial_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 10000;
    let start_temp: f64 = 100.;
    let end_temp: f64 = 0.1;
    let mut iter_count = 0;
    let mut progress;
    let mut temp;
    let start_time = time::elapsed_seconds();

    while time::elapsed_seconds() < time_limit {
        progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
        temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

        let (path_edges, path_verticies) = &paths[rnd::gen_range(0, paths.len())];
        // eprintln!("{:?}", path);
        // TODO: 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let prev = {
            let mut ret = vec![];
            for edge_index in path_edges {
                ret.push(state.when[*edge_index]);
            }
            ret
        };
        let next = rnd::gen_range(0, input.d);

        let mut new_score = state.score;
        for v in path_verticies {
            new_score -= calc_vertex_score(*v, &graph, &state);
        }

        for edge_index in path_edges {
            state.update_when(*edge_index, next);
        }
        for v in path_verticies {
            new_score += calc_vertex_score(*v, &graph, &state);
        }

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();
        if adopt && is_valid {
            state.score = new_score;
        } else {
            for i in 0..path_edges.len() {
                state.update_when(path_edges[i], prev[i]);
            }
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            if debug {
                writeln!(
                    score_progress_file,
                    "{},{:.2},{}",
                    state.score,
                    time::elapsed_seconds(),
                    calc_actual_score_slow(&input, &graph, &state),
                )
                .unwrap();
            }
            // eprintln!("{}, {:.2}", state.score, time::elapsed_seconds());
        }
    }
    eprintln!("[create_initial_state] iter_count: {}", iter_count);

    state
}

fn calc_cosine_similarity(to_pos: &Pos, from_pos: &Pos, to_pos2: &Pos, from_pos2: &Pos) -> f64 {
    let dy1 = to_pos.y - from_pos.y;
    let dx1 = to_pos.x - from_pos.x;

    let dy2 = to_pos2.y - from_pos2.y;
    let dx2 = to_pos2.x - from_pos2.x;

    let div = ((dy1 * dy1 + dx1 * dx1) as f64).sqrt() * ((dy2 * dy2 + dx2 * dx2) as f64).sqrt();
    let prod = (dy1 * dy2 + dx1 * dx2) as f64;

    prod / div
}
