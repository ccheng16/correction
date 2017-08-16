python feed_kenlm.py | ./kenlm/build/bin/lmplz -o 3 > zhwiki_trigram.arpa
./kenlm/build/bin/build_binary zhwiki_trigram.arpa zhwiki_trigram.klm
