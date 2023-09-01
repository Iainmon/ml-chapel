use IO, Json;

record Person {
  var name : string;
  var age : int;
  var d: domain(1,int) = {0..5};
}

var f = open("output.txt", ioMode.cw);

// configure 'w' to always write in JSON format
var w = f.writer(serializer=new JsonSerializer());

// writes:
// {"name":"Sam", "age":20}
var p = new Person("Sam", 20);
p.d = {1,2,3};
w.write(p);

f = open("output.txt", ioMode.r);

var r = f.reader(serializer=new JsonDeserializer());
var p2 = r.read(Person);
writeln(p2);