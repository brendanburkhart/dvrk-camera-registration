import math
import numpy as np

class DirectLinearTransformation:
    def __init__(self, dimension):
        self.dimension = dimension

        self.hspace_basis = DirectLinearTransformation._gen_hspace_basis(self.dimension)

    def _gen_hspace_basis(size):
        dimension = int(size*(size-1)/2)
        basis = []

        x, y = (0, 1)
        for i in range(dimension):
            H = np.zeros((size, size))
            H[x, y] = -1
            H[y, x] = 1

            y = y + 1
            if y >= size:
                x = x + 1
                y = x + 1

            basis.append(H)

        return basis

    def _flatten(self, M):
        return M.flatten('F')

    def _unflatten(self, v):
        return np.reshape(v, (self.dimension, self.dimension+1), 'F')

    def compute(self, x, y):
        n, d = x.shape
        assert d == self.dimension
        assert (n, d+1) == y.shape
        
        h = len(self.hspace_basis)
        B = np.zeros((n * h, self.dimension*(self.dimension+1)))

        for i in range(n):
            for j in range(h):
                H = self.hspace_basis[j]
                xH = np.matmul(np.transpose(x[i]), H).reshape(1,-1)
                r = np.matmul(np.transpose(xH), y[i].reshape(1,-1))
                B[i*h + j, :] = self._flatten(r)

        u, s, vh = np.linalg.svd(B)
        a = vh[len(s)-1, :]
        A = self._unflatten(a)
        return A

class CameraMatrix:
    def __init__(self, transformation):
        self.transformation = transformation

    def normalize(self):
        pass

    def rotation(self):
        return self.transformation[0:3, 0:3]

    def translation(self):
        return self.transformation[0:3, 3]


class CameraRegistration:
    def __init__(self, focal_length):
        self.focal_length = focal_length
        pass

    def estimate(self, camera_points, actual_points):
        dlt = DirectLinearTransformation(3)
        transformation = dlt.compute(camera_points, actual_points)
        # projection_inverse = np.zeros((3, 3))
        # projection_inverse[0, 0] = 1.0/self.focal_length
        # projection_inverse[1, 1] = 1.0/self.focal_length
        # projection_inverse[2, 2] = 1.0
        # transformation = np.matmul(projection_inverse, transformation)

        # scale so that rotation submatrix is a rotation, i.e. has determinant 1
        rotation = transformation[0:3, 0:3]
        det = np.linalg.det(rotation)
        scale = math.copysign(math.pow(1.0/math.fabs(det), 1/3), det)
        transformation = scale*transformation
        
        # add homogeneous-preserving row to create square matrix
        # not really necessary since we already have rotation + translation
        # but its nice to have camera matrix in standard form
        transformation = np.vstack([transformation, np.array([0, 0, 0, 1])])

        return CameraMatrix(transformation)
