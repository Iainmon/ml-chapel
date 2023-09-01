import Chai as chai;
import MNISTTester;

// var net = new chai.Network(
//     (
//         new chai.Conv(1,16,5,stride=2),
//         new chai.Conv(16,64,3),
//         new chai.MaxPool(),
//         new chai.SoftMax(10)
//     )
// );


// var net = new chai.Network(
//     (
//         new chai.Conv(1,8,7,stride=1),
//         new chai.Conv(8,12,5),
//         new chai.MaxPool(),
//         new chai.SoftMax(10)
//     )
// );

var net = new chai.Network(
    (
        new chai.Conv(1,8,7,stride=1),
        new chai.Conv(8,12,5),
        new chai.Conv(12,16,3),
        new chai.MaxPool(),
        new chai.SoftMax(10)
    )
);

config const numTrainImages = 20000;
config const numTestImages = 1000;

config const learnRate = 0.05; // 0.05;
config const batchSize = 10;
config const numEpochs = 40;


MNISTTester.train(
    network=net,
    numTrainImages=numTrainImages,
    numTestImages=numTestImages,
    learnRate=learnRate,
    batchSize=batchSize,
    numEpochs=numEpochs,
    savePath="../lib/models/demo_cnn" + net.signature() + ".model"
);