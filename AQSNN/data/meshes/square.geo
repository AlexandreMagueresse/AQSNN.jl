//+
Point(1) = {-1.0, -1.0, 0, 1.0};
Point(2) = {-1.0, +1.0, 0, 1.0};
Point(3) = {+1.0, +1.0, 0, 1.0};
Point(4) = {+1.0, -1.0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Point("boundary") = {1, 2, 3, 4};
//+
Physical Curve("boundary") = {1, 2, 3, 4};
