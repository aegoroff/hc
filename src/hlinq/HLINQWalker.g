tree grammar HLINQWalker;

options {
	tokenVocab	    = HLINQ;
    ASTLabelType    = pANTLR3_BASE_TREE;
    language	    = C;
    output=AST;
}

@header {
   #include "compiler.h"
#ifdef GTEST
  #include "displayError.h"
#endif
}
 
@apifuncs {
#ifdef GTEST
  RECOGNIZER->displayRecognitionError       = displayRecognitionErrorNew;
#endif
}

prog[apr_pool_t* root, ProgramOptions* po, const char* param]
@init { 
	InitProgram($po, $param, $root); 
}
	: statement*
	;
     
statement
@init {
        OpenStatement(RECOGNIZER->state); 
}
@after {
        CloseStatement();
}
    :   expr
    ;

expr:
	FOR 
	( { DefineQueryType(CtxTypeString); } expr_string 
	| { DefineQueryType(CtxTypeHash); } expr_hash  
	| { DefineQueryType(CtxTypeDir); } expr_dir
	| { DefineQueryType(CtxTypeFile); } expr_file  
	| { DefineQueryType(CtxTypeFile); } expr_file_analyze  
	)
	|
	expr_vardef
    ;

expr_vardef:
	^(VAR_DEF ID s=STRING) { RegisterVariable($ID.text->chars, $s.text->chars); }
	;

expr_string:
	^(HASH_STR {  RegisterIdentifier((pANTLR3_UINT8)"_s_"); } hash_clause source)
	;

expr_hash:
	^(BRUTE_FORCE id brute_force_clause let_clause? source)
	;

expr_dir
	: ^(HASH_DIR hash_clause id let_clause? where_clause? (WITHSUBS { SetRecursively(); })? source)
	| ^(HASH_DIR id let_clause? where_clause FIND { SetFindFiles(); } (WITHSUBS { SetRecursively(); })? source)
	;

expr_file
	: ^(HASH_FILE hash_clause id let_clause? source)
	| ^(HASH_FILE id let_clause source)
	;

expr_file_analyze
	: ^(ANALYZE_FILE id where_clause) 
	;
	
source : ID { SetSource($ID.text->chars, $ID); } | s=STRING { SetSource($s.text->chars, NULL); };
    
id : ID { RegisterIdentifier($ID.text->chars); };

attr_clause : ^(ATTR_REF ID attr) ;

attr : str_attr | int_attr ;

hash_clause
    : ^(ALG_REF ALG) {  SetHashAlgorithmIntoContext($ALG.text->chars);  }
    ;
    
brute_force_clause
	: CRACK hash_clause { SetBruteForce(); }
	;

let_clause
	: assign+
	;

where_clause
	: boolean_expression
    ;

boolean_expression returns [pANTLR3_UINT8 value, Attr code]
@init { 
	$value = NULL;
	$code = AttrUndefined;
}
	: ^(EQUAL l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpEq, $EQUAL, $l.value); }
	| ^(NOTEQUAL l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpNotEq, $NOTEQUAL, $l.value); }
	| ^(MATCH l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpMatch, $MATCH, $l.value); }
	| ^(NOTMATCH l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpNotMatch, $NOTMATCH, $l.value); }
	| ^(LE l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpLe, $LE, $l.value); }
	| ^(GE l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpGe, $GE, $l.value); }
	| ^(LEASSIGN l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpLeEq, $LEASSIGN, $l.value); }
	| ^(GEASSIGN l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpGeEq, $GEASSIGN, $l.value); }
	| ^(OR boolean_expression boolean_expression) { WhereClauseCond(CondOpOr, $OR); }
	| ^(AND boolean_expression boolean_expression) { WhereClauseCond(CondOpAnd, $AND); }
	| ^(NOT_OP boolean_expression) { WhereClauseCond(CondOpNot, $NOT_OP); }
	| ^(ATTR_REF ID boolean_expression) { CallAttiribute($ID.text->chars, $ID); }
	| STRING { $value = $STRING.text->chars; }
	| ID { $value = GetValue($ID.text->chars, $ID); }
	| INT { $value = $INT.text->chars; }
	| NAME_ATTR { $code = AttrName; $value = $NAME_ATTR.text->chars; }
	| PATH_ATTR { $code = AttrPath; $value = $PATH_ATTR.text->chars; }
	| ALG { $code = AttrHash; $value = $ALG.text->chars; }
	| SIZE_ATTR { $code = AttrSize; $value = $SIZE_ATTR.text->chars; }
	| LIMIT_ATTR { $code = AttrLimit; $value = $LIMIT_ATTR.text->chars; }
	| OFFSET_ATTR { $code = AttrOffset; $value = $OFFSET_ATTR.text->chars; }
	;

assign 
	:	^(ATTR_REF ID ^(ASSIGN_OP sa=str_attr s=STRING)) 
	{ 
		if(CallAttiribute($ID.text->chars, $ID)) {
			AssignAttribute($sa.code, $s.text->chars, $s, $sa.name);
		}
	}
	|
		^(ATTR_REF left=ID ^(ASSIGN_OP sa=str_attr right=ID)) 
	{ 
		if(CallAttiribute($left.text->chars, $left)) {
			AssignAttribute($sa.code, GetValue($right.text->chars, $right), $right, $sa.name);
		}
	}
	|	^(ATTR_REF ID ^(ASSIGN_OP ia=int_attr i=INT))
	{ 
		if(CallAttiribute($ID.text->chars, $ID)){
			AssignAttribute($ia.code, $i.text->chars, $i, NULL);
		}
	}
	;
 
str_attr returns[Attr code, pANTLR3_UINT8 name] 
@init { 
    $code = AttrUndefined; 
    $name = NULL;
}
	: NAME_ATTR  { $code = AttrName; }
	| PATH_ATTR  { $code = AttrPath; }
	| DICT_ATTR  { $code = AttrDict; }
	| ALG { $code = AttrHash; $name = $ALG.text->chars;  }
    ; 

int_attr returns[Attr code]
@init { $code = AttrUndefined; }
	: SIZE_ATTR { $code = AttrSize; } 
	| LIMIT_ATTR { $code = AttrLimit; } 
	| OFFSET_ATTR { $code = AttrOffset; } 
	| MIN_ATTR { $code = AttrMin; } 
	| MAX_ATTR { $code = AttrMax; } ; 

