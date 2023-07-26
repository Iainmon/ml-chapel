

record DTensor {
    type eltType = real;

    var _dimension: int;

    var _shapeDomain: domain(1,int);
    var shape: [_shapeDomain] int;

    var _dim_1_domain: domain(1,int) = {0..#0};
    var _dim_2_domain: domain(1,int) = {0..#0, 0..#0};
    var _dim_3_domain: domain(1,int) = {0..#0, 0..#0, 0..#0};

    var _dim_1_data: [_dim_1_domain] eltType;
    var _dim_2_data: [_dim_2_domain] eltType;
    var _dim_3_data: [_dim_3_domain] eltType;

    proc channel() ref : [?] eltType {
        select _dimension {
            case 1: return _dim_1_data;
            case 2: return _dim_2_data;
            case 3: return _dim_3_data;
        }
    }

    proc init(shape: int ...?d) where d < 4 {
        _dimension = d.size;
        _shapeDomain = {0..#shape};
        self.shape = shape;
        if _dimension == 1 {
            _dim_1_domain = {0..#shape[0]};
            _dim_1_data = new(_dim_1_domain, 0.0);
        } else if _dimension == 2 {
            _dim_2_domain = {0..#shape[0], 0..#shape[1]};
            _dim_2_data = new(_dim_2_domain, 0.0);
        } else if _dimension == 3 {
            _dim_3_domain = {0..#shape[0], 0..#shape[1], 0..#shape[2]};
            _dim_3_data = new(_dim_3_domain, 0.0);
        }
    }

}