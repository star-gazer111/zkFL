import { BarretenbergBackend } from '@noir-lang/backend_barretenberg';
import { Noir } from '@noir-lang/noir_js';
import circ1 from './circuit1/target/circuit1.json';
import circ2 from './circuit2/target/circuit2.json';
import axios from 'axios';

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

async function GenProofs(data) {
  display('epochs', `Running Epoch ${epochs++}`);
  let result_arr = [];

  // ========================================================
  let backend = new BarretenbergBackend(circ1);
  let noir = new Noir(circ1, backend);
  let arr1_1 = data[0][0]
  let arr1_2 = data[1][0]
  scale_arr(arr1_1)
  scale_arr(arr1_2)
  pre_process_arr(arr1_1)
  pre_process_arr(arr1_2)
  let input = { x: arr1_1, y: arr1_2 }
  console.log(input);
  display('logs', 'Generating proof [1]... ⌛')
  let proof = await noir.generateFinalProof(input);
  let result_arr_1 = [];
  for (let i = 0; i < 8; i++) {
    let a = Number((hexToDecimal(proof.publicInputs[i])));
    result_arr_1.push(a)
  }
  descale_arr(result_arr_1)
  display('logs', 'Generating proof [1]... ✅');
  display('results', proof.proof);
  display('logs', 'Verifying proof [1]... ⌛');
  let verification = await noir.verifyFinalProof(proof);
  if (verification) display('logs', 'Verifying proof [1]... ✅');

  // ========================================================
  backend = new BarretenbergBackend(circ1);
  noir = new Noir(circ1, backend);
  let arr2_1 = data[0][1]
  let arr2_2 = data[1][1]
  scale_arr(arr2_1)
  scale_arr(arr2_2)
  pre_process_arr(arr2_1)
  pre_process_arr(arr2_2)
  input = { x: arr2_1, y: arr2_2 }
  display('logs', 'Generating proof [2]... ⌛');
  proof = await noir.generateFinalProof(input);
  let result_arr_2 = [];
  for (let i = 0; i < 8; i++) {
    let a = Number((hexToDecimal(proof.publicInputs[i])));
    result_arr_2.push(a)
  }
  descale_arr(result_arr_2)
  display('logs', 'Generating proof [2]... ✅');
  display('results', proof.proof);
  display('logs', 'Verifying proof [2]... ⌛');
  verification = await noir.verifyFinalProof(proof);
  if (verification) display('logs', 'Verifying proof [2]... ✅');

  // ========================================================
  backend = new BarretenbergBackend(circ1);
  noir = new Noir(circ1, backend);
  let arr3_1 = data[0][2]
  let arr3_2 = data[1][2]
  scale_arr(arr3_1)
  scale_arr(arr3_2)
  pre_process_arr(arr3_1)
  pre_process_arr(arr3_2)
  input = { x: arr3_1, y: arr3_2 }
  display('logs', 'Generating proof [3]... ⌛');
  proof = await noir.generateFinalProof(input);
  let result_arr_3 = [];
  for (let i = 0; i < 8; i++) {
    let a = Number((hexToDecimal(proof.publicInputs[i])));
    result_arr_3.push(a)
  }
  descale_arr(result_arr_3)
  display('logs', 'Generating proof [3]... ✅');
  display('results', proof.proof);
  display('logs', 'Verifying proof [3]... ⌛');
  verification = await noir.verifyFinalProof(proof);
  if (verification) display('logs', 'Verifying proof [3]... ✅');

  // ===========================================================
  backend = new BarretenbergBackend(circ2);
  noir = new Noir(circ2, backend);
  let arr4_1 = data[0][3]
  let arr4_2 = data[1][3]
  scale_arr(arr4_1)
  scale_arr(arr4_2)
  pre_process_arr(arr4_1)
  pre_process_arr(arr4_2)
  input = { x: arr4_1[0], y: arr4_2[0] };
  display('logs', 'Generating proof [4]... ⌛');
  proof = await noir.generateFinalProof(input);
  let result_arr_4 = []
  let a = Number((hexToDecimal(proof.publicInputs[0])));
  result_arr_4.push(a)
  descale_arr(result_arr_4)
  display('logs', 'Generating proof [4]... ✅');
  display('results', proof.proof);
  display('logs', 'Verifying proof [4]... ⌛');
  verification = await noir.verifyFinalProof(proof);
  if (verification) display('logs', 'Verifying proof [4]... ✅');

  // ===========================================================
  result_arr.push(result_arr_1);
  result_arr.push(result_arr_2);
  result_arr.push(result_arr_3);
  result_arr.push(result_arr_4);
  return result_arr;
}

function display(container, msg) {
  const c = document.getElementById(container);
  const p = document.createElement('p');
  p.textContent = msg;
  c.appendChild(p);
}

let intervalId;

function getProofs () {
  console.log("Getting Proofs");
  axios.get('http://localhost:6969/api/v1/proofs/generate').then(async (response) => {
    console.log(response.data);
    if (response.data.run === true) {
      clearInterval(intervalId); 
      let Result = await GenProofs(response.data.data);
      axios.post('http://localhost:6969/api/v1/proofs/success', {
        result: Result
      }).then((response) => {
        console.log(response.data);
        intervalId = setInterval(getProofs, 10000);
      })
    } else {
      console.log("All Clients have not responded yet");
    }
  })
}

intervalId = setInterval(getProofs, 10000);
