// Require Statements
const express = require("express");
const app = express();
const port = 6969;
const morgan = require("morgan");
require('dotenv').config();
const cors = require("cors");

// CORS
app.use(cors());

// To send Data to the client
var bodyParser = require('body-parser');
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.use(morgan('dev'))

let target_clients = 2;
let clients = 0;
let callbacks = [];
let data = [];
let run_zk_proofs = false;

app.get("/api/v1/proofs/generate", (req, res) => {
    console.log(data);
    res.send({ run: run_zk_proofs, data: data });
})

app.post("/api/v1/proofs/success", async (req, res) => {
    Result = req.body.result;
    let arra1_to_send = []
    for (let i = 0; i < 8; i++) {
        arra1_to_send.push([Result[0][i]])
    }
    let arra2_to_send = Result[1]
    let arra3_to_send = [Result[2]]
    let arra4_to_send = Result[3]
    let array_to_send = JSON.stringify([arra1_to_send, arra2_to_send, arra3_to_send, arra4_to_send])
    callbacks.forEach(callback => callback({ result: array_to_send }));
    callbacks = [];
    clients = 0;
    run_zk_proofs = false;
    data = [];
    res.send({ status: "success" });
})

app.post("/api/v1/check", async (req, res) => {
    aggregated_array = req.body.params;
    aggregated_array[0] = JSON.parse(aggregated_array[0]);
    aggregated_array[1] = JSON.parse(aggregated_array[1]);
    aggregated_array[2] = JSON.parse(aggregated_array[2]);
    aggregated_array[3] = JSON.parse(aggregated_array[3]);
    let array1 = []
    for (let i = 0; i < 8; i++) {
        array1.push(aggregated_array[0][i][0]);
    }
    let array2 = aggregated_array[1]
    let array3 = aggregated_array[2][0]
    let array4 = aggregated_array[3]
    final_array = [array1, array2, array3, array4]
    data.push(final_array);
    clients++;
    if (clients === target_clients) {
        callbacks.push(response => res.send(response));
        run_zk_proofs = true;
        console.log("Running ZK Proofs");
    } else {
        callbacks.push(response => res.send(response));
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
