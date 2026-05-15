( AUTO GENERATED GRAMMAR FORTH WORDS )

( Sprawdzenie czy POS pasuje do kategorii )
: IS-NOUN?    ( pos_id -- bool ) 1 = ;
: IS-VERB?    ( pos_id -- bool ) 2 = ;
: IS-ADJ?     ( pos_id -- bool ) 3 = ;
: IS-ADV?     ( pos_id -- bool ) 4 = ;
: IS-PRONOUN? ( pos_id -- bool ) 5 = ;
: IS-PREP?    ( pos_id -- bool ) 6 = ;
: IS-CONJ?    ( pos_id -- bool ) 7 = ;
: IS-DET?     ( pos_id -- bool ) 11 = ;
: IS-AUX?     ( pos_id -- bool ) 16 = ;
: IS-MODAL?   ( pos_id -- bool ) 17 = ;

( Reguły gramatyczne — sprawdź czy sekwencja POS pasuje )
( S_BASIC: S -> NP VP )
( S_QUESTION: S -> AUX NP VP )
: MATCH-S_QUESTION ( -- ) IS-AUX? ;

( S_PASSIVE: S -> NP AUX V_PAST PRP NP )
: MATCH-S_PASSIVE ( -- ) IS-AUX? AND IS-PRP? ;

( S_COMPOUND: S -> S CNJ S )
: MATCH-S_COMPOUND ( -- ) IS-CNJ? ;

( NP_FULL: NP -> DET ADJ N )
: MATCH-NP_FULL ( -- ) IS-DET? AND IS-ADJ? AND IS-N? ;

( NP_SIMPLE: NP -> DET N )
: MATCH-NP_SIMPLE ( -- ) IS-DET? AND IS-N? ;

( NP_BARE: NP -> N )
: MATCH-NP_BARE ( -- ) IS-N? ;

( NP_PRONOUN: NP -> PRN )
: MATCH-NP_PRONOUN ( -- ) IS-PRN? ;

( NP_PROPER: NP -> NP_NAME )
( NP_PREP: NP -> NP PP )
( NP_MULTI_ADJ: NP -> DET ADJ ADJ N )
: MATCH-NP_MULTI_ADJ ( -- ) IS-DET? AND IS-ADJ? AND IS-ADJ? AND IS-N? ;

( VP_TRANS: VP -> V NP )
: MATCH-VP_TRANS ( -- ) IS-V? ;

( VP_INTRANS: VP -> V )
: MATCH-VP_INTRANS ( -- ) IS-V? ;

( VP_ADV: VP -> V ADV )
: MATCH-VP_ADV ( -- ) IS-V? AND IS-ADV? ;

( VP_PREP: VP -> V PP )
: MATCH-VP_PREP ( -- ) IS-V? ;

( VP_NP_PP: VP -> V NP PP )
: MATCH-VP_NP_PP ( -- ) IS-V? ;

( VP_MODAL: VP -> MOD V )
: MATCH-VP_MODAL ( -- ) IS-MOD? AND IS-V? ;

( VP_MODAL_NP: VP -> MOD V NP )
: MATCH-VP_MODAL_NP ( -- ) IS-MOD? AND IS-V? ;

( PP_BASIC: PP -> PRP NP )
: MATCH-PP_BASIC ( -- ) IS-PRP? ;

( ADJP_BASIC: ADJP -> ADJ )
: MATCH-ADJP_BASIC ( -- ) IS-ADJ? ;

( ADJP_ADV: ADJP -> ADV ADJ )
: MATCH-ADJP_ADV ( -- ) IS-ADV? AND IS-ADJ? ;

( ADVP_BASIC: ADVP -> ADV )
: MATCH-ADVP_BASIC ( -- ) IS-ADV? ;

( ADVP_PREP: ADVP -> PRP NP )
: MATCH-ADVP_PREP ( -- ) IS-PRP? ;

