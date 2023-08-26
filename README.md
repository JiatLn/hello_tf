## Hello tensorflow

Learn tensorflow with rust & python

### perpare model

- install dependencies

```bash
pip install tensorflow
```

- generate model

```bash
python pys/gen_model.py
```

- test model with python

```bash
python pys/infer_local.py
```

### rust grpc infer server

- define infer proto

- generate rust code

- implement server

- test server with rust

```bash
cargo run --bin infer_server
```
