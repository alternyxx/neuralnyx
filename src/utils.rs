use rand::Rng;

pub(crate) fn flatten2d(vec2d: &Vec<Vec<f32>>) -> Vec<f32> {
    vec2d.iter().flatten().copied().collect::<Vec<f32>>()
}

pub(crate) fn flatten3d(vec3d: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    vec3d.iter().flatten().flatten().copied().collect::<Vec<f32>>()
}

// maybe i should put this in the neural net struct? idrk.../
// also horrendous name
pub(crate) fn shuffle_with_correspondence<T>(
    batches: &mut Vec<T>,
    targets: &mut Vec<T>,
) -> Vec<usize> {
    let n_batches = batches.len(); // this could also be passed as an arg

    let mut rng = rand::rng();

    let mut shuffled_indices: Vec<usize> = (0..n_batches).collect();

    for i in 0..n_batches {
        let index_swap = rng.random_range(i..n_batches);

        batches.swap(i, index_swap);

        targets.swap(i, index_swap);

        shuffled_indices.swap(i, index_swap);
    }

    let mut shuffled_lookup = vec![0usize; n_batches];
    for (i, &shuffled_index) in shuffled_indices.iter().enumerate() {
        shuffled_lookup[shuffled_index] = i;
    }

    shuffled_lookup
}

// just infer that target_len is less than amount of padding required
// returns the amount of padding added just for convenience
pub (crate) fn pad2d(vec2d: &mut Vec<Vec<f32>>, target_len: usize){
    let vec1d_len = vec2d[0].len();

    for _ in 0..(target_len - vec2d.len()) {
        vec2d.push(vec![0.0; vec1d_len]);
    }
}