use core::slice;
use std::{ fs, fmt::Debug };
use tensorflow::{ Graph, SessionOptions, SavedModelBundle, Tensor, SessionRunArgs };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, tensorflow!");
    // load model
    let export_dir = "pys/resnet50";
    let mut graph = Graph::new();
    let opts = SessionOptions::new();
    let sm = SavedModelBundle::load(&opts, &["serve"], &mut graph, export_dir)?;

    // load request data
    let data = read_request("pys/request")?;

    // infer
    let mut step = SessionRunArgs::new();

    let output_op = graph.operation_by_name_required("StatefulPartitionedCall")?;
    step.add_target(&output_op);

    let input_op = graph.operation_by_name_required("serving_default_input_1")?;
    step.add_feed(&input_op, 0, &data);

    let output_token = step.request_fetch(&output_op, 0);

    sm.session.run(&mut step)?;

    let output_res: Tensor<f32> = step.fetch(output_token)?;

    println!("{:?}", output_res);

    Ok(())
}

fn read_request(path: &str) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let data = unsafe { slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
    Ok(Tensor::new(&[1, 224, 224, 3]).with_values(&data)?)
}
