use async_trait::async_trait;
use hello_tf::{infer_server, InferRequest, InferResponse};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use tokio::runtime::Builder;
use tonic::{transport::Server, Request, Response, Status};

struct InferImpl {
    model: SavedModelBundle,
    graph: Graph,
}

impl InferImpl {
    fn new(path: &str) -> Self {
        let mut graph = Graph::new();
        let opts = SessionOptions::new();
        let model = SavedModelBundle::load(&opts, &["serve"], &mut graph, path).unwrap();
        InferImpl { model, graph }
    }

    fn infer_logic(&self, request: Tensor<f32>) -> Tensor<f32> {
        let mut step = SessionRunArgs::new();

        let output_op = self
            .graph
            .operation_by_name_required("StatefulPartitionedCall")
            .unwrap();
        step.add_target(&output_op);

        let input_op = self
            .graph
            .operation_by_name_required("serving_default_input_1")
            .unwrap();
        step.add_feed(&input_op, 0, &request);

        let output_token = step.request_fetch(&output_op, 0);

        self.model.session.run(&mut step).unwrap();

        let output_res = step.fetch(output_token).unwrap();
        output_res
    }
}

#[async_trait]
impl infer_server::Infer for InferImpl {
    async fn infer(&self, req: Request<InferRequest>) -> Result<Response<InferResponse>, Status> {
        let req = req.into_inner();
        let request = Tensor::new(&req.shape).with_values(&req.data).unwrap();
        let output = self.infer_logic(request);
        let response = InferResponse {
            shape: output.dims().into(),
            data: output.to_vec(),
        };
        Ok(Response::new(response))
    }
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let addr = "127.0.0.1:5000".parse().unwrap();
        println!("Listening on {}", addr);
        let infer = InferImpl::new("pys/resnet50");
        let server = infer_server::InferServer::new(infer);

        Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    })
}
