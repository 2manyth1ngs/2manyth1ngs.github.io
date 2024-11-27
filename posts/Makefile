CSS = ../static/style.css
FLAG = --standalone --mathjax
SRC = src
TEMPLATE = template.yaml
RSS = ../atom.xml
MD = \
longterm_pessimism\
the_world_of_bit\
thought_of_mindset\
something_interesting

HTML = $(MD:%=%.html)

all: $(HTML)

%.html: $(SRC)/%.md $(TEMPLATE) $(CSS)
	@echo Converting $< to $@
	@pandoc $(TEMPLATE) $< -f markdown --css $(CSS) -t html $(FLAG) -o $@

clean:
	rm $(HTML)