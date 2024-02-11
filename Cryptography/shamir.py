"""
The Basic idea of Shamir's Secret Sharing is to split a secret into n shares in such a way that only k shares are required to reconstruct the secret.
"""

import random
from math import ceil
from decimal import Decimal

FIELD_SIZE = 10**5


def reconstruct_secret(shares):
    sums = 0

    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)

        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                prod *= Decimal(Decimal(xi)/(xi-xj))

        prod *= yj
        sums += Decimal(prod)

    return int(round(Decimal(sums), 0))


def polynom(x, coefficients):
    point = 0
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        point += x ** coefficient_index * coefficient_value
    return point


def coeff(t, secret):
    coeff = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coeff.append(secret)
    return coeff


def generate_shares(n, t, secret):
    coefficients = coeff(t, secret)
    shares = []

    for i in range(1, n+1):
        x = random.randrange(1, FIELD_SIZE)
        shares.append((x, polynom(x, coefficients)))

    return shares


if __name__ == '__main__':

    # (3,5) sharing scheme , pick a polynomial of degree 2 and pick 5 points on it. So the secret can be reconstructed from any 3 shares.
    t, n = 3, 5
    secret = 4242
    print(f'Original Secret: {secret}')
    shares = generate_shares(n, t, secret)
    # (x,y) points on the polynomial
    print(f'Shares: {", ".join(str(share) for share in shares)}')
    pool = random.sample(shares, t)
    print(f'Combining shares: {", ".join(str(share) for share in pool)}')
    print(f'Reconstructed secret: {reconstruct_secret(pool)}')
