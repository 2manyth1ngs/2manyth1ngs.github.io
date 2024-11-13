CSS = ../static/style.css
FLAG = --standalone --mathjax
SRC = src
TEMPLATE = template.yaml
RSS = ../atom.xml
MD = \
longterm_pessimism\
the_world_of_bit

HTML = $(MD:%=%.html)

all: $(HTML)

%.html: $(SRC)/%.md $(TEMPLATE) $(CSS)
	@echo Converting $< to $@
	@pandoc $(TEMPLATE) $< -f markdown --css $(CSS) -t html $(FLAG) -o $@

rss: rss_generator.py all
	@rm -f $(RSS)
	@echo Generating RSS feed...
	python3 ./rss_generator.py
	@echo Done!

clean:
	rm $(HTML)