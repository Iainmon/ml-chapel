class Conv:
    
    def __init__(self, num_filters):
        self.num_filters = num_filters
        
        #why divide by 9...Xavier initialization
        self.filters = np.random.randn(num_filters, 3, 3)/9
    
    def iterate_regions(self, image):
        #generates all possible 3*3 image regions using valid padding
        
        h,w = image.shape
        
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j
                
    def forward(self, input):
        self.last_input = input
        
        h,w = input.shape
        
        output = np.zeros((h-2, w-2, self.num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            filter_results = im_region * self.filters # A list of 3x3 matrices, 
                                                       # containing the hermitian product of im_region
                                                       # and each filter.
            sums = np.sum(filter_results, axis=(1,2)) # A 1d array/vector, where each ith element 
                                                      # is the sum of the components of the filter_results[i] matrix
            output[i, j] = sums
        return output
    def backprop(self, d_l_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_l_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i,j,f] * im_region

        #update filters
        self.filters -= learn_rate * d_l_d_filters

        return None

class MaxPool:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        
        new_h = h // 2
        new_w = w // 2
        
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j
    
    def forward(self, input):
        
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region,axis=(0,1))
            
        return output