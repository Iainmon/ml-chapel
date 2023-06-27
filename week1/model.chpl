use List;
use Map;
use nn;
use micrograd;

var epochCounter = 1;


var datasetIn = [
    [cnst(0),cnst(0),cnst(0)],
    [cnst(0),cnst(0),cnst(1)],
    [cnst(0),cnst(1),cnst(0)],
    [cnst(0),cnst(1),cnst(1)],
    [cnst(1),cnst(0),cnst(0)],
    [cnst(1),cnst(0),cnst(1)],
];
// var datasetOut = [
//     [cnst(0)],
//     [cnst(1)],
//     [cnst(2)],
//     [cnst(3)],
//     [cnst(4)],
//     [cnst(5)],
//     [cnst(6)],
// ];
var datasetOut = [
    [cnst(1),cnst(0),cnst(0),cnst(0),cnst(0),cnst(0)],
    [cnst(0),cnst(1),cnst(0),cnst(0),cnst(0),cnst(0)],
    [cnst(0),cnst(0),cnst(1),cnst(0),cnst(0),cnst(0)],
    [cnst(0),cnst(0),cnst(0),cnst(1),cnst(0),cnst(0)],
    [cnst(0),cnst(0),cnst(0),cnst(0),cnst(1),cnst(0)],
    [cnst(0),cnst(0),cnst(0),cnst(0),cnst(0),cnst(1)],
];
var dataset = datasetIn;
var trials = dataset;

proc epoch(mlp: Perceptron) {

    writeln("epoch: ", epochCounter);


    var resultsList = new list([0..#(datasetOut[0].size)] shared Expr);
    for d in dataset {
        resultsList.append(mlp.apply(d));
    }
    var results = resultsList.toArray(); // [i in dataset.domain] mlp.apply(dataset[i]);





    var totalCost: Expr = new shared Constant(0);

    for i in results.domain {
        var input = datasetIn[i];
        var expected = datasetOut[i];
        var result = results[i];
        var c = cost(result,expected);
        totalCost = totalCost + c;
    }


    for i in results.domain {
        var input = datasetIn[i];
        var result = results[i];
        var expected = datasetOut[i];
        writeln("Input: ", input.value(), " Expected: ", expected.value(), " Output: ", result.value(), " Local Cost: ", cost(result,expected).value());
    }

    var avgCost = totalCost / cnst(results.size);

    writeln("cost ", avgCost.value());


    var m = makeNudgeMap(avgCost);
    // writeln("delta ", m);

    epochCounter += 1;

    return (avgCost,m);
}


proc train(mlp: shared Perceptron, epochs: int) {
    for ep in 1..epochs {
        var (c,m) = epoch(mlp);
        // c.nudge(m);
        mlp.updateParams(m);
    }
}

const nin = datasetIn.size;
const nout = datasetOut[0].size;

var mlp = new shared Perceptron(3, [10,10,6]);


train(mlp,20000);
