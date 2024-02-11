function hexToDecimal(arr) {
    // Take the first 32 elements of the array
    const hexBytes = arr.slice(0, 32);

    // Convert the hexadecimal bytes to a single string
    const hexString = hexBytes.map(byte => byte.toString(16).padStart(2, '0')).join('');

    // Convert the hex string to decimal
    const decimalValue = BigInt(`0x${hexString}`);

    return decimalValue;
}

// Example array
const byteArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 10, 73, 73, 27, 54, 42]
console.log(byteArray.length);
const decimalValue = hexToDecimal(byteArray);
console.log(decimalValue.toString());