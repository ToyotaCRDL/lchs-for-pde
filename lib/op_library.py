import re
import sympy
import numpy as np
import scipy


from typing import Optional


def get_op_list(operator):

    op_list = []
    coeffs = []
    for term in operator.args:
        term = str(term).replace('**', 'x').split('*')
        
        try:
            coeff = float(term[0])
            i = 1
        except:
            if term[0][0] == '-':
                coeff = -1.0
                term[0] = term[0][1:]
            else:
                coeff = 1.0
            i = 0
        coeffs.append(coeff)

        expr = []
        for j in range(i, len(term)):
            s = term[j]
            m = re.match('[I,p,m,z,o,X,Y,Z]x[0-9]+', s)
            if m is None:
                expr.append(s)
            else:
                expr += [ s[0] ] * int(s[2:])
        op_list.append(''.join(expr))
    
    return op_list, coeffs


class OperatorList:

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def op_list(self):
        return self._op_list
    
    @property
    def coeffs(self):
        return self._coeffs

    def __init__(self, op_list, coeffs):
        self._op_list = op_list
        self._coeffs = coeffs
        self._num_qubits = len(op_list[0])
    
    def to_matrix(self, sparse=True):

        I = np.eye(2)
        s01 = np.array([[0., 1.],
                        [0., 0.]])
        s10 = np.array([[0., 0.],
                        [1., 0.]])
        s00 = np.array([[1., 0.],
                        [0., 0.]])
        s11 = np.array([[0., 0.],
                        [0., 1.]])
        X = np.array([[0., 1.],
                    [1., 0.]])
        Y = np.array([[0., -1j],
                    [1j, 0.]])
        Z = np.array([[1., 0.],
                    [0., -1.]])

        mat_dict = {'I': I, 'm': s01, 'p': s10, 'z': s00, 'o': s11}
        if sparse:
            for k, v in mat_dict.items():
                mat_dict[k] = scipy.sparse.csr_matrix(v)
            mat = scipy.sparse.csr_matrix((2**self._num_qubits,)*2, dtype=np.complex128)
        else:
            mat = np.zeros((2**self._num_qubits,)*2, dtype=np.complex128)

        for term, coeff in zip(self._op_list, self._coeffs):
            tmp = 1.
            for op in term:
                if sparse:
                    tmp = scipy.sparse.kron(tmp, mat_dict[op], format='csr')
                else:
                    tmp = np.kron(tmp, mat_dict[op])
            mat += coeff * tmp
        
        return mat

    def __add__(self, other):
        assert self._num_qubits == other._num_qubits, 'Operators to be added must have the same length'

        op_list = []
        coeffs = {}

        for op, coeff, in zip(self._op_list + other._op_list, self._coeffs + other.coeffs):
            if op not in op_list:
                op_list.append(op)
                coeffs[op] = coeff
            else:
                coeffs[op] += coeff
        
        return OperatorList([op_list[i] for i in np.argsort(op_list)], 
                            [coeffs[op] for op in np.sort(op_list)]).clip()

    def __sub__(self, other):
        assert self._num_qubits == other._num_qubits, 'Operators to be subtract must have the same length'
        op_list = []
        coeffs = {}

        for op, coeff, in zip(self._op_list + other._op_list, self._coeffs + [-1*coeff for coeff in other.coeffs]):
            if op not in op_list:
                op_list.append(op)
                coeffs[op] = coeff
            else:
                coeffs[op] += coeff
        
        return OperatorList([op_list[i] for i in np.argsort(op_list)], 
                            [coeffs[op] for op in np.sort(op_list)]).clip()

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, complex) or isinstance(other, int):
            return OperatorList([self._op_list[i] for i in np.argsort(self._op_list)], 
                                [other * self._coeffs[i] for i in np.argsort(self._op_list)]).clip()

        assert self._num_qubits == other._num_qubits, 'Operators to be multiplied must have the same length'

        op_list = []
        coeffs = {}
        for term_self, coeff_self in zip(self._op_list, self._coeffs):
            for term_other, coeff_other in zip(other._op_list, other._coeffs):
                c = []
                skip = False
                for a, b in zip(term_self, term_other):
                    # a x b
                    if a == 'I':
                        c.append(b)
                    elif b == 'I':
                        c.append(a)
                    elif a == 'm' and b == 'p':
                        c.append('z')
                    elif a == 'm' and b == 'o':
                        c.append('m')
                    elif a == 'p' and b == 'm':
                        c.append('o')
                    elif a == 'p' and b == 'z':
                        c.append('p')
                    elif a == 'z' and b == 'm':
                        c.append('m')
                    elif a == 'z' and b == 'z':
                        c.append('z')
                    elif a == 'o' and b == 'p':
                        c.append('p')
                    elif a == 'o' and b == 'o':
                        c.append(a)
                    else:
                        skip = True
    
                if not skip:
                    op = ''.join(c)
                    coeff = coeff_self * coeff_other
                    if op not in op_list:
                        op_list.append(op)
                        coeffs[op] = coeff
                    else:
                        coeffs[op] += coeff

        return OperatorList([op_list[i] for i in np.argsort(op_list)], 
                            [coeffs[op] for op in np.sort(op_list)]).clip()
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def clip(self, tol=1e-8):
        op_list = []
        coeffs = []
        for op, coeff in zip(self._op_list, self._coeffs):
            if np.abs(coeff) > tol:
                op_list.append(op)
                coeffs.append(coeff)
        return OperatorList([op_list[i] for i in np.argsort(op_list)], 
                            [coeffs[i] for i in np.argsort(op_list)])

    def concatenate(self, other):
        op_list = []
        coeffs = {}
        for op_self, coeff_self in zip(self._op_list, self._coeffs):
            for op_other, coeff_other in zip(other._op_list, other._coeffs):

                op = op_self + op_other
                coeff = coeff_self * coeff_other
                if op not in op_list:
                    op_list.append(op)
                    coeffs[op] = coeff
                else:
                    coeffs[op] += coeff
                    
        return OperatorList([op_list[i] for i in np.argsort(op_list)], 
                            [coeffs[op] for op in np.sort(op_list)]).clip()


class IdentityOperator(OperatorList):
    
    def __init__(self, num_qubits: int):
        super().__init__(['I'*num_qubits], [1.0])


class LaplacianOperator1d(OperatorList):
    
    def __init__(self, 
                num_qubits: int, 
                is_pauli: bool=False, 
                h: float=1.0, 
                periodic: bool=False,
                neumann: tuple[str]=()):
    
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits
        self._is_pauli = is_pauli
        self._h = h
        self._periodic = periodic
        self._neumann = neumann

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y)
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        # core of QTT for gradients w.r.t. x- and y-coordinates
        Cx = sympy.Matrix([
            [I, s01, s10],
            [0, s10, 0,],
            [0, 0, s01],
        ])

        # boundary cores
        C_left = sympy.Matrix([1, 0, 0]).T
        C_right = sympy.Matrix([-2, 1, 1])

        operator = 1 / h**2 * C_left * Cx**num_qubits * C_right
        operator = sympy.expand(operator[0])

        if self._periodic:
            operator += 1 / h**2 * s10 ** num_qubits + 1 / h**2 * s01 ** num_qubits
        if 'left' in self._neumann:
            operator += 1 / h**2 * s00**num_qubits_x
        if 'right' in self._neumann:
            operator += 1 / h**2 * s11**num_qubits_x
        

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class LaplacianOperator2d(OperatorList):

    @property
    def num_qubits_x(self):
        return self._num_qubits_x
    
    @property
    def num_qubits_y(self):
        return self._num_qubits_y
    
    def __init__(self, 
                num_qubits_x: int, 
                num_qubits_y: int=None,
                is_pauli: bool=False, 
                h: float=1.0, 
                periodic: tuple[str]=(),
                neumann: tuple[str]=()):
    
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x

        if isinstance(periodic, str):
            periodic = tuple(periodic)

        num_qubits = num_qubits_x + num_qubits_y
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits_x
        self._num_qubits_y = num_qubits_y
        self._is_pauli = is_pauli
        self._h = h
        self._periodic = periodic
        self._neumann = neumann

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y) 
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)   
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        # core of QTT for gradients w.r.t. x- and y-coordinates
        Cx = sympy.Matrix([
            [I, s01, s10, 0, 0, 0],
            [0, s10, 0, 0, 0, 0],
            [0, 0, s01, 0, 0, 0],
            [0, 0, 0, I, 0, 0],
            [0, 0, 0, 0, I, 0],
            [0, 0, 0, 0, 0, I]
        ])
        Cy = sympy.Matrix([
            [I, 0, 0, 0, 0, 0],
            [0, I, 0, 0, 0, 0],
            [0, 0, I, 0, 0, 0],
            [0, 0, 0, I, s01, s10],
            [0, 0, 0, 0, s10, 0],
            [0, 0, 0, 0, 0, s01]
            ])

        # boundary cores
        C_left = sympy.Matrix([1, 0, 0, 1, 0, 0]).T
        C_right = sympy.Matrix([-2, 1, 1, -2, 1, 1])

        # contraction of QTT to generate qubit operator
        operator = 1 / h**2 * C_left * Cy**num_qubits_y * Cx**num_qubits_x * C_right
        operator = sympy.expand(operator[0])

        if 'x' in self._periodic:
            operator += 1 / h**2 * I**num_qubits_y * s10**num_qubits_x + 1 / h**2 * I**num_qubits_y * s01**num_qubits_x
        if 'y' in self._periodic:
            operator += 1 / h**2 * s10**num_qubits_y * I**num_qubits_x + 1 / h**2 * s01**num_qubits_y * I**num_qubits_x

        if 'left' in self._neumann:
            operator += 1 / h**2 * I**num_qubits_y * s00**num_qubits_x
        if 'right' in self._neumann:
            operator += 1 / h**2 * I**num_qubits_y * s11**num_qubits_x
        if 'bottom' in self._neumann:
            operator += 1 / h**2 * s00**num_qubits_y * I**num_qubits_x
        if 'top' in self._neumann:
            operator += 1 / h**2 * s11**num_qubits_y * I**num_qubits_x

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class DifferentialOperator1d(OperatorList):
    
    def __init__(self, 
                num_qubits: int, 
                is_pauli=False, 
                diff_type: str='central', 
                h: float=1.0, 
                periodic: bool=False):
    
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits
        self._is_pauli = is_pauli
        self._diff_type = diff_type
        self._h = h
        self._periodic = periodic

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y)
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        if diff_type == 'central':
            # core of QTT for gradients w.r.t. x--coordinates
            Cx = sympy.Matrix([
                [I, s01, s10],
                [0, s10, 0,],
                [0, 0, s01],
            ])

            # boundary cores
            C_left = sympy.Matrix([0.5, 0, 0]).T
            C_right = sympy.Matrix([0, 1, -1])
        
        elif diff_type == 'forward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s01],
                [0, s10,]
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([-1, 1])
        
        elif diff_type == 'backward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s10],
                [0, s01],
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([1, -1])
        
        else:
            raise NotImplementedError()

        operator = 1 / h * C_left * Cx**num_qubits * C_right
        operator = sympy.expand(operator[0])

        if self._periodic:
            if diff_type == 'central':
                operator += 0.5 / h * s10 ** num_qubits - 0.5 / h * s01 ** num_qubits

            elif diff_type == 'forward':
                operator += 1 / h * s10 ** num_qubits
            
            elif diff_type == 'backward':
                operator -= 1 / h * s01 ** num_qubits

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class DifferentialOperator(OperatorList):

    def __init__(self, 
                num_qubits_x: int, 
                num_qubits_y: Optional[int]=None,
                dim: int=1, 
                axis: str='x',
                diff_type: str='central', 
                h: float=1.0, 
                periodic: tuple[str]=()):

        self._diff_type = diff_type
        self._h = h
        self._periodic = periodic

        if diff_type not in ['central', 'forward', 'backward']:
            raise NotImplementedError()

        if isinstance(periodic, str):
            periodic = tuple(periodic)

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x
            op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type=diff_type, h=h, periodic=periodic)
            op_list = op._op_list
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            self._num_qubits = self._num_qubits_x + self._num_qubits_y
            if axis == 'x':
                if isinstance(periodic, tuple):
                    periodic_x = 'x' in periodic
                elif isinstance(periodic, bool):
                    periodic_x = periodic
                op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type=diff_type, h=h, periodic=periodic_x)
                op_list = ['I'*self._num_qubits_y + op for op in op._op_list]
                
            elif axis == 'y':
                if isinstance(periodic, tuple):
                    periodic_y = 'y' in periodic
                elif isinstance(periodic, bool):
                    periodic_y = periodic
                op = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type=diff_type, h=h, periodic=periodic_y)
                op_list = [op + 'I'*self._num_qubits_x for op in op._op_list]
                
        else:
            raise NotImplementedError()        
        
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [op._coeffs[i] for i in np.argsort(op_list)]


class ShiftOperator1d(OperatorList):
    
    def __init__(self, 
                num_qubits: int, 
                is_pauli=False, 
                direction: str='forward',
                periodic: bool=False):
    
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits
        self._is_pauli = is_pauli
        self._direction = direction
        self._periodic = periodic

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y)
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        if direction == 'backward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s01],
                [0, s10,]
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([0, 1])
        
        elif direction == 'forward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s10],
                [0, s01],
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([0, 1])
        
        else:
            raise NotImplementedError()

        operator = C_left * Cx**num_qubits * C_right
        operator = sympy.expand(operator[0])

        if self._periodic:
            if direction == 'forward':
                operator += s10 ** num_qubits
            
            elif direction == 'backward':
                operator += s01 ** num_qubits

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class ShiftOperator(OperatorList):

    def __init__(self, 
                num_qubits_x: int, 
                num_qubits_y: Optional[int]=None,
                dim: int=1, 
                axis: str='x',
                direction: str='forward', 
                periodic: tuple[str]=()):

        self._direction = direction
        self._periodic = periodic

        if direction not in ['forward', 'backward']:
            raise NotImplementedError()

        if isinstance(periodic, str):
            periodic = tuple(periodic)

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x
            op = ShiftOperator1d(self._num_qubits_x, is_pauli=False, direction=direction, periodic=periodic)
            op_list = op._op_list
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            self._num_qubits = self._num_qubits_x + self._num_qubits_y
            if axis == 'x':
                if isinstance(periodic, tuple):
                    periodic_x = 'x' in periodic
                elif isinstance(periodic, bool):
                    periodic_x = periodic
                op = ShiftOperator1d(self._num_qubits_x, is_pauli=False, direction=direction, periodic=periodic_x)
                op_list = ['I'*self._num_qubits_y + op for op in op._op_list]
                
            elif axis == 'y':
                if isinstance(periodic, tuple):
                    periodic_y = 'y' in periodic
                elif isinstance(periodic, bool):
                    periodic_y = periodic
                op = ShiftOperator1d(self._num_qubits_y, is_pauli=False, direction=direction, periodic=periodic_y)
                op_list = [op + 'I'*self._num_qubits_x for op in op._op_list]
                
        else:
            raise NotImplementedError()        
        
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [op._coeffs[i] for i in np.argsort(op_list)]


class WaveEquationEvolution(OperatorList):
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def diff_type(self):
        return self._diff_type
    
    @property
    def h(self):
        return self._h

    def __init__(self, 
                num_qubits_x: int, 
                num_qubits_y: Optional[int]=None,
                dim: int=1, 
                diff_type: str='central', 
                h: float=1.0, 
                periodic: tuple[str]=(),
                compact: bool=True):

        if isinstance(periodic, str):
            periodic = tuple(periodic)

        self._dim = dim
        self._h = h
        self._periodic = periodic
        self._compact = compact

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x + 1
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            n_ancilla = 1 if compact else 2
            self._num_qubits = self._num_qubits_x + self._num_qubits_y + n_ancilla
        
        else:
            raise NotImplementedError()
        
        if diff_type not in ['central', 'forward', 'backward']:
            raise NotImplementedError()            

        self._diff_type = diff_type
        
        op_list = []
        coeffs = {}
        
        if dim == 1:
            if diff_type == 'central':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
            
            elif diff_type == 'forward':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                
                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
            
            elif diff_type == 'backward':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                
                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
                        
        elif dim == 2:
            if diff_type == 'central':
                if isinstance(periodic, tuple):
                    periodic_x = 'x' in periodic
                    periodic_y = 'y' in periodic
                elif isinstance(periodic, bool):
                    periodic_x = periodic
                    periodic_y = periodic
                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic_x)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='central', h=h, periodic=periodic_y)

                if self._compact:
                    diff_op_list = ['I'*self._num_qubits_y + op for op in diff_op_x._op_list] \
                                + [op + 'I'*self._num_qubits_x for op in diff_op_y._op_list]
                    diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'm' + op not in op_list:
                            op_list.append('m' + op)
                            coeffs['m' + op] = coeff
                        else:
                            coeffs['m' + op] += coeff
                    
                    diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]
    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'p' + op not in op_list:
                            op_list.append('p' + op)
                            coeffs['p' + op] = -coeff
                        else:
                            coeffs['p' + op] -= coeff
                
                else:
                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'm' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'm' + 'm' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'p' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'p' + 'p' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff
                    
            elif diff_type == 'forward':
                if isinstance(periodic, tuple):
                    periodic_x = 'x' in periodic
                    periodic_y = 'y' in periodic
                elif isinstance(periodic, bool):
                    periodic_x = periodic
                    periodic_y = periodic
                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic_x)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='forward', h=h, periodic=periodic_y)

                if self._compact:
                    diff_op_list = ['I'*self._num_qubits_y + op for op in diff_op_x._op_list] \
                                + [op + 'I'*self._num_qubits_x for op in diff_op_y._op_list]
                    diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'm' + op not in op_list:
                            op_list.append('m' + op)
                            coeffs['m' + op] = coeff
                        else:
                            coeffs['m' + op] += coeff
                else:
                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'm' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'm' + 'm' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic_x)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='backward', h=h, periodic=periodic_y)

                if self._compact:
                    diff_op_list = ['I'*self._num_qubits_y + op for op in diff_op_x._op_list] \
                                + [op + 'I'*self._num_qubits_x for op in diff_op_y._op_list]
                    diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]
    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'p' + op not in op_list:
                            op_list.append('p' + op)
                            coeffs['p' + op] = -coeff
                        else:
                            coeffs['p' + op] -= coeff
                else:
                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'p' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'p' + 'p' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff

            elif diff_type == 'backward':
                if isinstance(periodic, tuple):
                    periodic_x = 'x' in periodic
                    periodic_y = 'y' in periodic
                elif isinstance(periodic, bool):
                    periodic_x = periodic
                    periodic_y = periodic
                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic_x)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='backward', h=h, periodic=periodic_y)

                if self._compact:
                    diff_op_list = ['I'*self._num_qubits_y + op for op in diff_op_x._op_list] \
                                + [op + 'I'*self._num_qubits_x for op in diff_op_y._op_list]
                    diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'm' + op not in op_list:
                            op_list.append('m' + op)
                            coeffs['m' + op] = coeff
                        else:
                            coeffs['m' + op] += coeff
                else:
                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'm' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'm' + 'm' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = coeff
                        else:
                            coeffs[op_new] += coeff

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic_x)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='forward', h=h, periodic=periodic_y)

                if self._compact:
                    diff_op_list = ['I'*self._num_qubits_y + op for op in diff_op_x._op_list] \
                                + [op + 'I'*self._num_qubits_x for op in diff_op_y._op_list]
                    diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]
    
                    for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                        if 'p' + op not in op_list:
                            op_list.append('p' + op)
                            coeffs['p' + op] = -coeff
                        else:
                            coeffs['p' + op] -= coeff
                else:
                    for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                        op_new = 'p' + 'z' + 'I'*self._num_qubits_y + op
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff

                    for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                        op_new = 'p' + 'p' + op + 'I'*self._num_qubits_x
                        if op_new not in op_list:
                            op_list.append(op_new)
                            coeffs[op_new] = -coeff
                        else:
                            coeffs[op_new] -= coeff
        
        else:                
            raise NotImplementedError()

        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[op] for op in np.sort(op_list)]


class HeatEquationEvolution(OperatorList):
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def diff_type(self):
        return self._diff_type
    
    @property
    def h(self):
        return self._h
    
    def __init__(self, 
                num_qubits_x: int, 
                num_qubits_y: Optional[int]=None,
                dim: int=1, 
                h: float=1.0, 
                periodic: tuple[str]=(),
                kappa: float=1.0):

        if isinstance(periodic, str):
            periodic = tuple(periodic)

        self._dim = dim
        self._h = h
        self._periodic = periodic
        self._kappa = kappa

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            self._num_qubits = self._num_qubits_x + self._num_qubits_y
        
        else:
            raise NotImplementedError()
                
        op_list = []
        coeffs = {}
        
        if dim == 1:

            diff_op = LaplacianOperator1d(self._num_qubits_x, is_pauli=False, h=h, periodic=periodic)

            for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                if op not in op_list:
                    op_list.append(op)
                    coeffs[op] = 1j * self._kappa * coeff
                else:
                    coeffs[op] += 1j * self._kappa * coeff
            
        elif dim == 2:
            if isinstance(periodic, tuple):
                periodic_x = 'x' in periodic
                periodic_y = 'y' in periodic
            elif isinstance(periodic, bool):
                periodic_x = periodic
                periodic_y = periodic
            diff_op_x = LaplacianOperator1d(self._num_qubits_x, is_pauli=False, h=h, periodic=periodic_x)
            diff_op_y = LaplacianOperator1d(self._num_qubits_y, is_pauli=False, h=h, periodic=periodic_y)
            
            for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                if 'I'*self._num_qubits_y + op not in op_list:
                    op_list.append('I'*self._num_qubits_y + op)
                    coeffs['I'*self._num_qubits_y + op] = 1j * self._kappa * coeff
                else:
                    coeffs['I'*self._num_qubits_y + op] += 1j * self._kappa * coeff
            
            for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                if op + 'I'*self._num_qubits_x not in op_list:
                    op_list.append(op + 'I'*self._num_qubits_x)
                    coeffs[op + 'I'*self._num_qubits_x] = 1j * self._kappa * coeff
                else:
                    coeffs[op + 'I'*self._num_qubits_x] += 1j * self._kappa * coeff
        
        else:
            raise NotImplementedError()
            
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[op] for op in np.sort(op_list)]
