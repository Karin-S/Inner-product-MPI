inner_products : inner_products.c
	mpicc -lm -o $@ $^

clean:
	rm inner_products
