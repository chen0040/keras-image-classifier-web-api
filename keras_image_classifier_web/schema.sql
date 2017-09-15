-- drop table if exists projects;
create table if not exists projects (
  id integer primary key autoincrement,
  name text not null,
  'description' text not null,
  token text not null
);