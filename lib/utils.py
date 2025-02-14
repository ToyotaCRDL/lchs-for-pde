import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer
from qiskit.circuit.library import CXGate


import scikit_tt.tensor_train as tt
from scikit_tt import TT


def mps2circuit(mps: TT, D:int=1) -> QuantumCircuit:

    '''
    convert right canonical MPS to qiskit's Quantumcircuit object

    mps: TT object of scikit-tt whose cores must be 4-dimensional numpy.ndarray with the size of (a, i, 1, b),
           where a and b are the bond dimensions and i is the physical dimension.
           i must be 2.
    '''

    qc = QuantumCircuit(len(mps.cores))
    decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='U')
    mps_l = mps.copy()

    for l in range(D):

        if max(mps_l.ranks) == 1:
            print("MPS has been disentangled. No additional layers are required.")
            break

        qc_tmp = QuantumCircuit(len(mps.cores))

        mps_l_trunc = mps_l.copy()
        mps_l_trunc.ortho(max_rank=2)
        cores = (1. / mps_l_trunc.norm() * mps_l_trunc).cores
        mpd_unitaries = []
        for i, core in enumerate(cores):
    
            if i == 0:
                unitary = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
                unitary[:, 0] = core.flatten()
    
                if np.abs(np.linalg.norm(unitary[:, 0]) - 1) > 1e-9:
                    raise ValueError('MPS is not in the canonical form.')
                
                tmp = unitary[:, 1] - np.vdot(unitary[:, 0], unitary[:, 1]) * unitary[:, 0]
                unitary[:, 1] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 2] - np.vdot(unitary[:, 0], unitary[:, 2]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 2]) * unitary[:, 1]
                unitary[:, 2] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 3] - np.vdot(unitary[:, 0], unitary[:, 3]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 3]) * unitary[:, 1] \
                                    - np.vdot(unitary[:, 2], unitary[:, 3]) * unitary[:, 2]
                unitary[:, 3] = tmp / np.linalg.norm(tmp)
    
                qc_tmp.compose(decomposer(unitary), qubits=[qc.num_qubits - 2, qc.num_qubits - 1], inplace=True)
                mpd_unitaries.append(unitary.transpose().conjugate())
            
            elif i < len(mps.cores) - 1:
    
                unitary = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
                unitary[:, 0] = core[0, :, :, :].flatten()
                unitary[:, 2] = core[1, :, :, :].flatten()
    
                if np.abs(np.vdot(unitary[:, 0], unitary[:, 2])) > 1e-9 or np.abs(np.linalg.norm(unitary[:, 0]) - 1) > 1e-9 or np.abs(np.linalg.norm(unitary[:, 2]) - 1) > 1e-9:
                    raise ValueError('MPS is not in the canonical form.')
                
                tmp = unitary[:, 1] - np.vdot(unitary[:, 0], unitary[:, 1]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 2], unitary[:, 1]) * unitary[:, 2]
                unitary[:, 1] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 3] - np.vdot(unitary[:, 0], unitary[:, 3]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 3]) * unitary[:, 1] \
                                    - np.vdot(unitary[:, 2], unitary[:, 3]) * unitary[:, 2]
                unitary[:, 3] = tmp / np.linalg.norm(tmp)
    
                qc_tmp.compose(decomposer(unitary), qubits=[qc.num_qubits - 2 - i, qc.num_qubits - 1 - i], inplace=True)
                mpd_unitaries.append(unitary.transpose().conjugate())
            
            else:
                unitary = core.reshape((2, 2)).transpose()
                theta, phi, lam = OneQubitEulerDecomposer().angles(unitary)
                qc_tmp.u(theta, phi, lam, 0)
                mpd_unitaries.append(unitary.transpose().conjugate())

        qc.compose(qc_tmp, front=True, inplace=True)

        for i, mpd_unitary in enumerate(reversed(mpd_unitaries)):

            if i == 0:
                mpd = tt.eye([2]*(len(mps.cores)-1)).concatenate(TT(mpd_unitary.reshape((2, 2))))

            elif i == 1:
                mpd = tt.eye([2]*(len(mps.cores)-2)).concatenate(TT(mpd_unitary.reshape((2, 2, 2, 2))))

            elif i < len(mps.cores) - 1:
                mpd = tt.eye([2]*(len(mps.cores)-i-1)).concatenate(TT(mpd_unitary.reshape((2, 2, 2, 2)))).concatenate(tt.eye([2]*(i-1)))

            else:
                mpd = TT(mpd_unitary.reshape((2, 2, 2, 2))).concatenate(tt.eye([2]*(i-1)))
                
            mps_l = mpd.dot(mps_l)
            mps_l.ortho(threshold=1e-12)
                
    return qc


def resolve_conflicts(bits1, bits2, coeff):
    # resolve the bits counted both in bits1 and bits2
    # by modifying the string bits1
    bits_diff = []
    bits1_arbit = []
    bits2_arbit = []
    for i, (b1, b2) in enumerate(zip(bits1, bits2)):
        if b1 == '-' or b2 == '-':
            if b1 == '-':
                bits1_arbit.append(i)
            if b2 == '-':
                bits2_arbit.append(i)
        elif b1 != b2:
            bits_diff.append(i)

    # check duplication
    if len(bits_diff) == 0:
        for i in bits1_arbit:
            if i not in bits2_arbit:
                if bits2[i] == '0':
                    b_mod = '1'
                else:
                    b_mod = '0'
                bits1 = bits1[:i] + b_mod + bits1[i+1:]
                return bits1, bits2, coeff

        for i in bits2_arbit:
            if i not in bits1_arbit:
                if bits1[i] == '0':
                    b_mod = '1'
                else:
                    b_mod = '0'
                bits1 = bits1[:i] + b_mod + bits1[i+1:]
                return bits1, bits2, -1*coeff

        return bits1, bits2, 0

    return bits1, bits2, coeff

def custom_structure2d(file, check=True, add_identity=0):
    str_dict = {}

    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.split()
        bits, val = line[0], line[1]
        
        # check duplicate bits
        coeff = float(val)
        if check:
            for key in str_dict.keys():
                bits, _, coeff = resolve_conflicts(bits, key, coeff)
                if coeff == 0:
                    break

        if coeff:
            str_dict[bits] = coeff

    structure_op = {}
    for bits, coeff in str_dict.items():
        bits = bits.replace('-', 'I').replace('0', 'z').replace('1', 'o')

        if add_identity:
            bits = bits[:len(bits)//2] + 'I'*add_identity + bits[len(bits)//2:] + 'I'*add_identity
        structure_op[bits] = float(coeff)

    return structure_op