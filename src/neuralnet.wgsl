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
${storage}
var<private> softmax_outputs: array<f32, n_outputs>;

@compute @workgroup_size(${batch_size})
fn forward_pass(@builtin(global_invocation_id) id: vec3u) { 
    // i made this a template and grouped it because 
    // otherwise it causes a stack overflow
${forward}

    softmax_activation();

    outputs.costs[id.x] = categorial_cross_entropy(
        targets[id.x],
    );

${backpropagate}
}

// tanh is a built-in function

fn dtanh(al: f32) -> f32 {
    return 1.0 - (al * al);
}

fn relu(zl: f32) -> f32 {
    return max(0.0, zl);
}

fn drelu(al: f32) -> f32 {
    return select(0.0, 1.0, al > 0.0);
}

fn sigmoid(zl: f32) -> f32 {
    return 1.0 / (1.0 + exp(-zl));
}

fn dsigmoid(al: f32) -> f32 {
    return al * (1.0 - al);
}

fn softmax_activation() {
    // find the highest
    var highest = al${n_al}[0];
    for (var i = 1; i < n_outputs; i++) {
        highest = max(highest, al${n_al}[i]);
    }
    
    // calculate e_i^zl
    var sum = 1.0e-20;
    for (var i = 0; i < n_outputs; i++) {
        let tmp = exp(al${n_al}[i] - highest);
        softmax_outputs[i] = tmp;
        sum += tmp; 
    }

    // e_i^zl / sum(e^zl)
    for (var i = 0; i < n_outputs; i++) {
        softmax_outputs[i] /= sum;
    }
}

fn categorial_cross_entropy(
    expected_outputs_i: array<f32, n_outputs>, 
) -> f32 {
    var cost = 0.0;
    for (var i = 0; i < n_outputs; i++) {
        cost += expected_outputs_i[i]
            * log(max(softmax_outputs[i], 1.0e-7));
    }
    return -cost;
}