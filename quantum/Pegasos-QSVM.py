from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC

from qiskit_machine_learning.kernels import QuantumKernel
import time

from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend

def main(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):

    train_features = kwargs.pop("train_features")
    train_labels = kwargs.pop("train_labels")
    test_features = kwargs.pop("test_features")
    test_labels = kwargs.pop("test_labels")
    n_features = kwargs.pop("n_features")
    n_reps = kwargs.pop("n_reps", 3)
    entanglement_type = kwargs.pop("entanglement_type", "circular")

    # number of qubits is equal to the number of features
    num_qubits = n_features

    # number of steps performed during the training procedure
    tau = 100

    # regularization parameter
    C = 1000

    start_time = time.time()

    fm = ZFeatureMap(feature_dimension=num_qubits, reps=1)
    #fm = ZZFeatureMap(feature_dimension=n_features, reps=n_reps, entanglement=entanglement_type)
    kernel = QuantumKernel(feature_map=fm, quantum_instance=backend)

    pegasos_qsvc = PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=tau)

    # training
    pegasos_qsvc.fit(train_features, train_labels)

    # testing
    pegasos_score = pegasos_qsvc.score(test_features, test_labels)

    finish_time = time.time() - start_time

    return pegasos_score, finish_time