all: \
	index.md \
	0_intro.ipynb \
	1_create_edge_container.ipynb \
	workspace/measure_pi.ipynb 

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_edge_container.ipynb \
	workspace/measure_pi.ipynb 

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_edge_container.md \
		snippets/measure_pi.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_edge_container.ipynb: snippets/create_edge_container.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_edge_container.md \
                -o 1_create_edge_container.ipynb  
	sed -i 's/attachment://g' 1_create_edge_container.ipynb

workspace/measure_pi.ipynb: snippets/measure_pi.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/measure_pi.md \
                -o workspace/measure_pi.ipynb  
	sed -i 's/attachment://g' workspace/measure_pi.ipynb
