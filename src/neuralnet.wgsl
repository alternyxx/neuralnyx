// custom wgsl file allowing javascript string interpolation :questionable:
// and since i have that why use pipeline constants :shrug:
// there are examples provided below the template used to provide an example
// of what the passed values could be, for exact values, check the generate_wgsl() in neuralnet.rs
const n_inputs = ${n_inputs};
const batch_size = ${batch_size};
const n_outputs = ${n_outputs};

struct Weights {
    ${i_weights}    // ie, weights0: array<array<f32, 12>, 9>
}

struct Biases {
    ${i_biases}    // ie, biases0: array<f32, 12>
}

struct Outputs {
    costs: array<f32, batch_size>,
    grad_weights: array<Weights, batch_size>,
    grad_biases: array<Biases, batch_size>,
}

@group(0) @binding(0) var<storage> X: array<array<f32, n_inputs>, batch_size>;
@group(0) @binding(1) var<storage> weights: Weights;
@group(0) @binding(2) var<storage> biases: Biases;
@group(0) @binding(3) var<storage> targets: array<array<f32, n_outputs>, batch_size>;
@group(0) @binding(4) var<storage, read_write> outputs: Outputs;

@compute @workgroup_size(${batch_size})
fn forward_pass(@builtin(global_invocation_id) id: vec3u) { 
    // i made this a template and grouped it because 
    // otherwise it causes a stack overflow
    ${forward}
    // var al0 = array<f32, 9>();
    // for (var i = 0; i < 9; i += 1) {
    //     for (var j = 0; j < 9; j += 1) {
    //         al0[i] += weights.weights0[i][j] * X[id.x][j];
    //     }
    //     al0[i] += biases.biases0[i];
    // };

    let softmax_outputs = softmax_activation(al${n_al});
    
    outputs.costs[id.x] = categorial_cross_entropy(
        targets[id.x],
        softmax_outputs,
    );

    // outputs.grad_weights.weights0[8][8] = 1.0;

    for (var i = 0; i < 9; i++) {
        let dal = (softmax_outputs[i] - targets[id.x][i]);

        for (var j = 0; j < 9; j++) {
            outputs.grad_weights[id.x].weights0[i][j] = X[id.x][j] * dal;
        }

        outputs.grad_biases[id.x].biases0[i] = dal;
    }
}

// tanh is alr available

fn relu(zl: f32) -> f32 {
    return max(0.0, zl);
}

fn sigmiod(zl: f32) -> f32 {
    return 1 / (1 + exp(-zl));
}

// O(n^2) :sob: there is easily a better way but im too lazy :3
fn softmax_activation(zl: array<f32, n_outputs>) -> array<f32, n_outputs> {
    var softmax_outputs = array<f32, n_outputs>();

    // calculate e_i^zl
    var sum = 1.0e-20;
    for (var i = 0; i < n_outputs; i += 1) {
        let tmp = exp(zl[i]);
        softmax_outputs[i] = tmp;
        sum += tmp; 
    }

    // e_i^zl / sum(e^zl)
    for (var i = 0; i < n_outputs; i += 1) {
        softmax_outputs[i] /= sum;
    }

    return softmax_outputs;
}

fn categorial_cross_entropy(
    expected_outputs_i: array<f32, n_outputs>, 
    softmax_outputs: array<f32, n_outputs>,
) -> f32 {
    var cost = 0.0;
    for (var i = 0; i < n_outputs; i += 1) {
        cost += expected_outputs_i[i] 
            * log(clamp(softmax_outputs[i], 1.0e-5, 1.0));
    }
    return -cost;
}