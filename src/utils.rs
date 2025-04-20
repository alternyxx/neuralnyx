pub(crate) fn flatten2d(vec2d: &Vec<Vec<f32>>) -> Vec<f32> {
    vec2d.iter().flatten().copied().collect::<Vec<f32>>()
}

pub(crate) fn flatten3d(vec3d: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    vec3d.iter().flatten().flatten().copied().collect::<Vec<f32>>()
}

// a lazy implementation that only shuffles dimensions, not between them 
// if that even makes sense
// pub(crate) fn shuffle3d(vec3d: &mut Vec<Vec<Vec<f32>>>) {
//     let mut rng = rand::rng();

//     for vec2d in vec3d.iter_mut() {
//         for vec in vec2d.iter_mut() {
//             vec.shuffle(&mut rng);
//         }
//         vec2d.shuffle(&mut rng);
//     }
//     vec3d.shuffle(&mut rng);
// }

// pub(crate) fn shuffle(
//     batches: &mut Vec<Vec<Vec<f32>>>,
//     &mut <Vec<Vec<Vec<f32>>>
// ) {
//     let rng = rand::rng();
// 
// 
// }