CSS = ../static/style.css
FLAG = --standalone --mathjax
SRC = src
TEMPLATE = template.yaml
RSS = ../atom.xml
MD = \
longterm_pessimism\
the_world_of_bit\
thought_of_mindset\
something_interesting\
miserable_thought\
fall_in_these_days\
media_of_memories\
power_of_rationality\
curse_of_writing\
diary\
i_need_change\

HTML = $(MD:%=%.html)

all: $(HTML)

%.html: $(SRC)/%.md $(TEMPLATE) $(CSS)
	@echo Converting $< to $@
	@pandoc $(TEMPLATE) $< -f markdown --css $(CSS) -t html $(FLAG) -o $@

clean:
	rm $(HTML)