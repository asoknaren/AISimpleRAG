-- public.qa_records definition

-- Drop table

-- DROP TABLE public.qa_records;

CREATE TABLE public.qa_records (
	id serial4 NOT NULL,
	question text NOT NULL,
	answer text NOT NULL,
	question_embedding public.vector NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT qa_records_pkey PRIMARY KEY (id)
);