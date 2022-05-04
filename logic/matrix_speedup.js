export class Matrix extends Array {
    /**
     * Add new methods to two dimentional array
     * @param {Array<Array<Number>>} body budy of target matrix
     */
    constructor(rows, cols) {
        super(rows);

        for (let i = 0; i < rows; i++) {
            this[i] = Array(cols);
        }

        this.rows = rows;
        this.cols = cols;
    }

    static fill(matrix, func) {
        let rows = matrix.rows, cols = matrix.cols;
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                matrix[i][j] = func(i, j);
    }

    static checkMul(matrix, multer) {
        return matrix.cols == multer.rows;
    }

    static mul(matrix, multer) {
        let rows = matrix.rows, cols = matrix.cols, multerRows = multer.rows;

        let res = new Matrix(rows, multer.cols);
        let localRes; //int

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < multerRows; j++) {
                localRes = 0;
                for (let c = 0; c < cols; c++)
                    localRes += matrix[i][c] * multer[c][j];
                res[i][j] = localRes;
            }
        }
        return res;
    }
}

export class Dense extends Array {
    constructor(cols) {
        super(cols);
    }

    static mul(dense, matrix) {
        // matrix rows == dense cols
        let rows = dense.length, cols = matrix.cols;
        let res = new Dense(cols), localRes;

        for (let j = 0; j < cols; j++) {
            localRes = 0;
            for (let i = 0; i < rows; i++)
                localRes += dense[i] * matrix[i][j];
            res[j] = localRes;
        }

        return res;
    }
}