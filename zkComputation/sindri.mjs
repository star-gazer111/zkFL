// Use dynamic imports for ESM and node REPL compatibility, not necessary otherwise.
const axios = (await import("axios")).default;
const FormData = (await import("form-data")).default;
const fs = (await import("fs")).default;
const process = (await import("process")).default;
const tar = (await import("tar")).default;

const SINDRI_API_KEY = process.env.SINDRI_API_KEY

axios.defaults.baseURL = "https://sindri.app/api/v1";
axios.defaults.headers.common["Authorization"] = `Bearer ${SINDRI_API_KEY}`;
axios.defaults.validateStatus = (status) => status >= 200 && status < 300;


let epochs = 0;

function scale_arr(arr) {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = arr[i] * 1e16;
  }
}

function pre_process_arr(arr) {
  for (let i = 0; i < arr.length; i++) {  
    arr[i] = parseFloat((arr[i].toString().split('.'))[0]);
    if (arr[i] < 0) {
      arr[i] = -arr[i];
    }
    if (arr[i] % 2 === 1) {
      arr[i] = arr[i] + 1;
    }
  }
}

function hexToDecimal(arr) {
  const hexBytes = [];
  for (let i = 0; i < 32; i++) {
    hexBytes.push(arr[i])
  }
  const hexString = hexBytes.map(byte => {
    const byteValue = byte >= 0 && byte <= 255 ? byte : 0;
    return byteValue.toString(16).padStart(2, '0');
  }).join('');
  const decimalValue = BigInt(`0x${hexString}`);
  return decimalValue;
}

function descale_arr(arr) {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = arr[i] / 1e16;
  }
}


// Create a new circuit.
const formData = new FormData();
formData.append(
  "files",
  tar.c({ gzip: true, sync: true }, ["circuit1/"]).read(),
  {
    filename: "compress.tar.gz",
  },
);

const createResponse = await axios.post(
  "/circuit1/create",
  formData,
);
const circuitId = createResponse.data.circuit_id;
console.log("Circuit ID:", circuitId);

// Poll for completed status.
let startTime = Date.now();
let circuitDetailResponse;
while (true) {
  circuitDetailResponse = await axios.get(`/circuit1/${circuitId}/detail`, {
    params: { include_verification_key: false },
  });
  const { status } = circuitDetailResponse.data;
  const elapsedSeconds = ((Date.now() - startTime) / 1000).toFixed(1);
  if (status === "Ready") {
    console.log(`Polling succeeded after ${elapsedSeconds} seconds.`);
    break;
  } else if (status === "Failed") {
    throw new Error(
      `Polling failed after ${elapsedSeconds} seconds: ${circuitDetailResponse.data.error}.`,
    );
  } else if (Date.now() - startTime > 30 * 60 * 1000) {
    throw new Error("Timed out after 30 minutes.");
  }
  await new Promise((resolve) => setTimeout(resolve, 1000));
}
console.log("Circuit Detail:");
console.log(circuitDetailResponse.data);
const package_name = circuitDetailResponse.data.nargo_package_name;

let arr2_1 = data[0][1]
let arr2_2 = data[1][1]
// Generate a new proof and poll for completion.
const proofInput = "edgelist = [222, 331, 152, 294, 43, 270, 313, 278, 210, 383, 74, 22, 250, 317, 66, 169, 214, 385, 49, 337, 134, 5, 91, 1, 41, 299, 394, 160, 182, 299]";
const proveResponse = await axios.post(`/circuit1/${circuitId}/prove`, {
  proof_input: proofInput,
});
const proofId = proveResponse.data.proof_id;
console.log("Proof ID:", proofId);
startTime = Date.now();
let proofDetailResponse;
while (true) {
  proofDetailResponse = await axios.get(`/proof/${proofId}/detail`);
  const { status } = proofDetailResponse.data;
  const elapsedSeconds = ((Date.now() - startTime) / 1000).toFixed(1);
  if (status === "Ready") {
    console.log(`Polling succeeded after ${elapsedSeconds} seconds.`);
    break;
  } else if (status === "Failed") {
    throw new Error(
      `Polling failed after ${elapsedSeconds} seconds: ${proofDetailResponse.data.error}.`,
    );
  } else if (Date.now() - startTime > 30 * 60 * 1000) {
    throw new Error("Timed out after 30 minutes.");
  }
  await new Promise((resolve) => setTimeout(resolve, 1000));
}
console.log("Proof Output:");
console.log(proofDetailResponse.data.proof);
console.log("Public Output:");
console.log(proofDetailResponse.data.public);

// Create circuits/proofs if it does not exist
const proof_dir = "./circuit1/proofs";
if (!fs.existsSync(proof_dir)){
  fs.mkdirSync(proof_dir);
}

// Save the proof in appropriate Nargo-recognizable file
fs.writeFileSync(
  "circuits/proofs/"+package_name+".proof",
  String(proofDetailResponse.data.proof["proof"]),
);

// Save the public data in appropriate Nargo-recognizable file
fs.writeFileSync(
  "circuits/Verifier.toml",
  String(proofDetailResponse.data.public["Verifier.toml"]),
);
