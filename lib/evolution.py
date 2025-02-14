import cmath
from typing import List


import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


import scikit_tt.tensor_train as tt
import scikit_tt.solvers.sle as sle
from scikit_tt import TT


from lib import OperatorList
from .utils import mps2circuit


class HamiltonianSimulation():
    
    def __init__(self,
                operator: OperatorList,
                crop_thres: float=1e-6):
        self._operator = operator
        self._num_qubits_system = operator.num_qubits
        self._crop_thres = crop_thres
        
        self._dg_dict = {'p': 'm', 'm': 'p', 'I': 'I', 'z': 'z', 'o': 'o'}
        self._separate_operator(crop_thres)
        
        if len(self._op_h_list) > 0:
            self._with_hermitian = True
        else:
            self._with_hermitian = False
        
        if len(self._op_a_list) > 0:
            self._with_antihermitian = True
        else:
            self._with_antihermitian = False
    
    def _separate_operator(self, crop_thres):

        op_h_list = []
        op_a_list = []
        coeffs_h = {}
        coeffs_a = {}
        for op, coeff, in zip(self._operator.op_list, self._operator.coeffs):

            op_dg = ''.join([self._dg_dict[sop] for sop in op])

            if op not in op_h_list:
                op_h_list.append(op)
                coeffs_h[op] = 0.5 * coeff
            else:
                coeffs_h[op] += 0.5 * coeff
            
            if op_dg not in op_h_list:
                op_h_list.append(op_dg)
                coeffs_h[op_dg] = 0.5 * np.conjugate(coeff)
            else:
                coeffs_h[op_dg] += 0.5 * np.conjugate(coeff)
            
            if op not in op_a_list:
                op_a_list.append(op)
                coeffs_a[op] = 1j * 0.5 * coeff
            else:
                coeffs_a[op] += 1j * 0.5 * coeff
            
            if op_dg not in op_a_list:
                op_a_list.append(op_dg)
                coeffs_a[op_dg] = -1j * 0.5 * np.conjugate(coeff)
            else:
                coeffs_a[op_dg] += -1j * 0.5 * np.conjugate(coeff)
        
        self._op_h_list = []
        self._coeffs_h = []
        for op in np.sort(op_h_list):
            if np.abs(coeffs_h[op]) > crop_thres:
                self._op_h_list.append(op)
                self._coeffs_h.append(coeffs_h[op])

        self._op_a_list = []
        self._coeffs_a = []
        for op in np.sort(op_a_list):   
            if np.abs(coeffs_a[op]) > crop_thres:
                self._op_a_list.append(op)
                self._coeffs_a.append(coeffs_a[op])


class LinearCombinationHamiltonianSimulation(HamiltonianSimulation):
    
    def __init__(self,
                operator: OperatorList,
                num_qubits_lcu: int=3,
                lsb_pos: int=0,
                crop_thres: float=1e-6):
        
        super().__init__(operator, crop_thres)
        self._num_qubits_lcu = num_qubits_lcu # number of total auxiliary qubits for linear combination of unitaries
        self._lsb_pos = lsb_pos
        
        if self._with_antihermitian:
            self._lcu_state_preparation_circ = self._get_lcu_state_preparation_circ()
            self._num_qubits = self._num_qubits_system + self._num_qubits_lcu
        else:
            self._lcu_state_preparation_circ = None
            self._num_qubits = self._num_qubits_system

    @property
    def kernel_mps(self):
        if hasattr(self, '_kernel_mps'):
            return self._kernel_mps
        else:
            return None
    
    def _get_lcu_state_preparation_circ(self, r_newton=10, r_inv=2, num_layers: int=1):
        n_int = self._num_qubits_lcu + self._lsb_pos - 1
        left_core = np.array([-2**n_int] + [2**k for k in reversed(range(self._lsb_pos, n_int))])
        right_core = np.ones(self._num_qubits_lcu)
        core = np.zeros((self._num_qubits_lcu, 2, 1, self._num_qubits_lcu))
        for k in range(self._num_qubits_lcu):
            core[k, :, 0, k] = np.ones(2)

        core_list = []
        for k in range(self._num_qubits_lcu):
            core_tmp = core.copy()
            core_tmp[k, 0, 0, k] = 0
            if k == 0:
                core_tmp = np.einsum('i, ijkl->jkl', left_core, core_tmp)[None, :, :, :]
            elif k == self._num_qubits_lcu - 1:
                core_tmp = np.einsum('ijkl, l->ijk', core_tmp, right_core)[:, :, :, None]
            core_list.append(core_tmp)
        
        n_lcu = self._num_qubits_lcu
        k_mps = TT(core_list)
        tt_ones = tt.ones(row_dims=[2]*n_lcu, col_dims=[1]*n_lcu, ranks=1)
        k_mps_diag = k_mps.diag(t=k_mps, diag_list=list(range(n_lcu)))
        target_mps = k_mps_diag.dot(k_mps) + tt_ones
        mps = tt.rand(tt_ones.row_dims, tt_ones.col_dims)
        for it in range(50):
            mps_diag = mps.diag(t=mps, diag_list=list(range(n_lcu)))
            rhs = mps_diag.dot(mps) + (-1)*target_mps

            # print("iter: {}, residue: {}".format(it, rhs.norm()))
            if rhs.norm() < 1e-6:
                break

            grad = -2*mps_diag
            dmps = sle.mals(operator=grad, 
                            initial_guess=tt.rand(tt_ones.row_dims, tt_ones.col_dims), 
                            right_hand_side=rhs, 
                            repeats=5, 
                            max_rank=r_newton)
            mps = mps + dmps
            mps.ortho(max_rank=r_newton)

        k_inv_mps = sle.mals(operator=mps.diag(t=mps, diag_list=list(range(n_lcu))), 
                            initial_guess=tt.rand(tt_ones.row_dims, tt_ones.col_dims), 
                            right_hand_side=2**(0.5*self._lsb_pos) / np.sqrt(np.pi) * tt_ones, 
                            repeats=5, 
                            max_rank=r_inv)
        k_inv_mps = 1. / k_inv_mps.norm() * k_inv_mps
        k_inv_mps.ortho(max_rank=r_inv)
        self._kernel_mps = k_inv_mps

        return mps2circuit(k_inv_mps, D=num_layers)
    
    def _get_evolve_circ_part(self, dt, op_list, coeffs, *, control=False, barrier=False):
        meas_circ_dict = {}
        coeffs_dict = {}
        num_qubits = self._num_qubits_system + self._num_qubits_lcu * control
        circ = QuantumCircuit(num_qubits)

        for term_op, coeff in zip(op_list, coeffs):
            # Listing qubits where term_op acts on non-trivially.
            qubits_s01 = []
            qubits_s10 = []
            qubits_s00 = []
            qubits_s11 = []

            for i, op in enumerate(reversed(term_op)): # The "term op" is "reversed" to fit the endian to that of qiskit.
                if op == 'm': 
                    qubits_s01.append(i)
                elif op == 'p':
                    qubits_s10.append(i)
                elif op == 'z':
                    qubits_s00.append(i)
                elif op == 'o':
                    qubits_s11.append(i)
            
            is_idendity = False
            if term_op == 'I'*len(term_op):
                is_idendity = True
            
            # Since we assume that the operator is symmetric matrix, 
            # we only focus on the term where the left most non-trivial operator is s01
            # to construct the circuit.
            if len(qubits_s01) > 0:
                if len(qubits_s10) == 0 or qubits_s01[-1] > qubits_s10[-1]:
                    
                    q_controls = []
                    
                    # Rotating basis to the Bell basis
                    for i in reversed(qubits_s01[:-1]):
                        circ.cx(qubits_s01[-1], i)
                        circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                        q_controls.append(circ.qubits[i])
                    for i in reversed(qubits_s10):
                        circ.cx(qubits_s01[-1], i)
                        q_controls.append(circ.qubits[i])
                    for i in reversed(qubits_s00):
                        circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                        q_controls.append(circ.qubits[i])
                    for i in reversed(qubits_s11):
                        q_controls.append(circ.qubits[i])

                    lam = cmath.phase(coeff)
                    gam = abs(coeff)

                    circ.p(lam, qubits_s01[-1])
                    circ.h(qubits_s01[-1])

                    if barrier:
                        circ.barrier()
                    
                    # rotation
                    if not control:
                        if len(q_controls) > 0:
                            circ.mcrz(2*dt*gam, q_controls, circ.qubits[qubits_s01[-1]])
                        else:
                            circ.rz(2*dt*gam, circ.qubits[qubits_s01[-1]])
                    else:
                        for i in range(self._num_qubits_lcu):
                            if len(q_controls) > 0:
                                circ.mcrz(2*dt[i]*gam, [circ.qubits[-i-1]] + q_controls, circ.qubits[qubits_s01[-1]])
                            else:
                                circ.crz(2*dt[i]*gam, circ.qubits[-i-1], circ.qubits[qubits_s01[-1]])

                    if barrier:
                        circ.barrier()

                    # uncomputation
                    circ.h(qubits_s01[-1])
                    circ.p(-lam, qubits_s01[-1])

                    for i in qubits_s00:
                        circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                    for i in qubits_s10:
                        circ.cx(qubits_s01[-1], i)
                    for i in qubits_s01[:-1]:
                        circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                        circ.cx(qubits_s01[-1], i)
                    
                    if barrier:
                        circ.barrier()
                        circ.barrier()

            if is_idendity and control:
                for i in range(self._num_qubits_lcu):
                    circ.p(-dt[i]*abs(coeff), circ.qubits[-i-1])

        return circ
    
    def get_evolve_circ(self,
                        dt: float,
                        barrier: bool=False,
                        measure: bool=True):

        q_sys = QuantumRegister(self._num_qubits_system, r'q_{sys}')
        if self._with_antihermitian:
            q_anc = QuantumRegister(self._num_qubits_lcu, r'q_{anc}')
            circ = QuantumCircuit(q_sys, q_anc)
        else:
            q_anc = None
            circ = QuantumCircuit(q_sys)
        

        # evolution from Hermitian part
        if self._with_hermitian:
            circ_h = self._get_evolve_circ_part(dt, self._op_h_list, self._coeffs_h, barrier=barrier)
            circ = circ.compose(circ_h, qubits=circ.qubits[:self._num_qubits_system])

        # linear combination of unitaries from anti-Hermitian part
        if self._with_antihermitian:
            n_int = self._num_qubits_lcu + self._lsb_pos - 1
            k_list = [-2**n_int] + [2**k for k in reversed(range(self._lsb_pos, n_int))]
            dt_list = [dt*k for k in k_list]
            circ = circ.compose(self._lcu_state_preparation_circ, qubits=circ.qubits[-self._num_qubits_lcu:])
            circ_a = self._get_evolve_circ_part(dt_list, self._op_a_list, self._coeffs_a, control=True, barrier=barrier)
            circ = circ.compose(circ_a, qubits=circ.qubits)
            circ = circ.compose(self._lcu_state_preparation_circ.inverse(), qubits=circ.qubits[-self._num_qubits_lcu:])

            if measure:
                c_reg = ClassicalRegister(self._num_qubits_lcu)
                circ.add_register(c_reg)
                circ.measure(q_anc, c_reg)

        return circ